"""
PDF-Chatbot-V2 — Chat Engine (Phase 3)
======================================
Interactive CLI & single-query client for querying indexed PDFs via:
  1. Embed query with nomic-embed-text-v1.5 (using "search_query: " prefix)
  2. Hybrid Retrieval:
     - Dense Vector Search (ChromaDB top-K)
     - Sparse BM25 Search (rank-bm25 top-K)
     - Reciprocal Rank Fusion (RRF) to merge candidate lists
  3. Cross-Encoder Re-ranking (mxbai-rerank-xsmall-v1)
  4. Contiguous chunk merging & context structuring
  5. Stream response from Ollama (qwen3:8b, 32k context)
  6. Source citation & conversation memory

Handles all document types: instructions, guides, manuals, reports,
code specifications, and general informational text.

Usage:
    python query.py                          # Interactive CLI
    python query.py --query "What is X?"     # Single-shot query
    python query.py --top-k 8                # Context chunks

Commands (in interactive mode):
    /clear    — Reset conversation memory
    /sources  — Show sources from last retrieval
    /count    — Show total documents in vector store
    /help     — Show available commands
    /quit     — Exit
"""

import argparse
import json
import logging
import pickle
import time

import chromadb
import requests
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

import config
from utils import code_aware_tokenize, detect_device, setup_logging

# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI technical architect and systems assistant for the Aurelis Command Center (ACC) ecosystem.
Your job is to provide thorough, logically rigorous, and actionable answers by synthesizing the provided PDF documentation
with deep cloud-native engineering and platform knowledge.

Product & Architecture Context:
- Aurelis Command Center (ACC) is an enterprise cloud-native network management platform deployed across Kubernetes
  clusters using Helm charts, containerized microservices (web UI, topology/inventory, alarm analyzer, authentication,
  device adapters), message queues (Kafka), and databases (MariaDB/MaxScale, Redis, PostgreSQL/MongoDB).
- The ACC WebUI is a client-side Single Page Application (SPA) communicating via REST APIs and WebSockets to underlying
  Kubernetes backend service pods.
- Installation Architecture:
  * Software Prerequisites: Supported Linux (Ubuntu 22.04/24.04, RHEL 8.x/9.x), Python 3.8-3.12, pip3, and Ansible (v2.9 to v2.16).
  * Online Guided Installer: Relies on host-level Ansible and upstream repository access. If errors regarding missing
    Ansible packages occur, it indicates missing OS prerequisites, missing pip dependencies, or lack of upstream repository access.
    Resolution requires installing Ansible via the OS package manager (`apt install -y ansible python3-pip` or `dnf install -y ansible python3-pip`)
    or using the airgapped Offline Installer package (`aurelis_install.bin`) which bundles all dependencies.
- Geo-Redundancy & High Availability Architecture:
  * In Active-Standby / Disaster Recovery (Cold or Warm Standby), the Active site runs all operational microservice pods.
  * In the Standby site, only data/database replication components (MariaDB MaxScale replication, Cross-Cluster Replication)
    are active. The main application workload pods remain in a dormant / not running / scaled-down status by design to prevent
    split-brain and write conflicts.
  * Application pods on the standby site transition to active/running state only during a planned or disaster switchover.
  * Seeing application pods not running on the standby site is normal and expected design behavior if replication is healthy
    (`maxscale_adm --verify-datacenter-replication` returns OK) and no geo-redundancy replication alarms exist.

Core Diagnostic & Response Rules:
1. End-to-End Holistic Triage (Full-Picture System Understanding):
   - When a user reports that a WebUI feature, view (e.g. Network Views, ONT View, Alarm Analyzer), or service fails
     to open, freezes, or returns errors, connect the frontend symptom with the entire system stack:
     a. Infrastructure / Pod Health: Suggest checking Kubernetes pod statuses (`kubectl get pods -n <namespace>`) to
        identify crashing or unready pods (e.g., `CrashLoopBackOff`, `OOMKilled`, `ImagePullBackOff`, `Pending`).
     b. Backend Microservice Logs: Suggest inspecting container logs for the relevant service
        (`kubectl logs -n <namespace> -l app=<service-name>` or checking journald/docker logs).
     c. Browser / Client Diagnostics: Check Browser Developer Tools (F12 Console for JavaScript errors; Network tab for
        HTTP 500/502/504 errors or failed WebSocket `/ws` connections), and try clearing session/browser cache.
     d. Authorization / RBAC: Verify that the user's role profile in ACC User Management has view/read permissions.
     f. Documented Alternative Routes: Mention alternative deep-link navigation routes if documented in ACC (e.g.,
        accessing topology via ONT View or Alarm Analyzer popups).
2. Logical Precondition & Sanity Validation (CRITICAL):
   - NEVER tell a user to navigate or click inside a view, menu, or page that they reported is inaccessible, broken,
     or failing to open (e.g., if "Network Views is not opening", do NOT tell them to "Click device UNI in Network Views").
3. Documentation Grounding & Page Citations:
   - Ground answers in the provided Context whenever relevant. Accurately cite specific source files and page numbers
     (e.g., "[Source: Simpleinstallation.pdf, Page 2]").
   - Distinguish clearly between standard online installations and airgapped offline installer procedures.
4. Intelligent Augmentation without Hallucination:
   - When the user asks about runtime failures, Linux configurations, Docker/Kubernetes diagnostics, or networking not
     explicitly in the indexed PDF text, provide accurate, standard cloud-native engineering practices.
   - Do NOT invent fake product menus or nonexistent software flags.
5. Multi-turn Isolation:
   - Treat each new user prompt as a distinct inquiry. Answer the new question directly without echoing past turns."""

QUERY_TEMPLATE = """## Context (Retrieved from indexed PDFs for Current Question)
{context}

## Previous Conversation (Context Only — Do NOT repeat past answers)
{history}

## Current Question (FOCUS EXCLUSIVELY ON THIS)
{question}

## Answer to Current Question:"""


# ──────────────────────────────────────────────
# Conversation Memory
# ──────────────────────────────────────────────


class ConversationMemory:
    """
    Sliding-window conversation memory buffer.
    Keeps the last N turns (question + answer pairs) for follow-up context.
    """

    def __init__(self, max_turns: int = config.MAX_MEMORY_TURNS):
        self.max_turns = max_turns
        self.turns: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]

    def add_turn(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.turns.append({"role": role, "content": content})
        # Trim to keep only last N turns (each turn = 2 messages)
        max_messages = self.max_turns * 2
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]

    def format_history(self) -> str:
        """Format conversation history concisely to avoid topic hijacking."""
        if not self.turns:
            return "(No previous conversation)"

        lines = []
        for msg in self.turns:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            text = msg["content"]
            # Keep assistant turn summaries concise to prevent hallucination looping
            if msg["role"] == "assistant" and len(text) > 200:
                text = text[:200].rstrip() + "... [details omitted]"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def clear(self):
        """Reset the conversation memory."""
        self.turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self.turns) // 2


# ──────────────────────────────────────────────
# Hybrid Retriever (ChromaDB + BM25 + RRF)
# ──────────────────────────────────────────────


class Retriever:
    """
    Handles hybrid retrieval combining dense vector search (ChromaDB)
    and sparse keyword search (BM25) via Reciprocal Rank Fusion (RRF),
    followed by Cross-Encoder re-ranking.
    """

    def __init__(self, device: str = None):
        self.logger = logging.getLogger("pdf_chatbot_v2.retriever")
        self.device = device or detect_device()

        # Load embedding model
        self.logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(
            config.EMBEDDING_MODEL, device=self.device, trust_remote_code=True
        )

        # Connect to ChromaDB
        self.logger.info(f"Connecting to vector store: {config.VECTORSTORE_DIR}")
        self.client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))

        try:
            self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
            count = self.collection.count()
            self.logger.info(
                f"Vector store loaded — {count} documents in collection '{config.COLLECTION_NAME}'"
            )
            if count == 0:
                self.logger.warning("⚠ Vector store is empty! Run 'python ingest.py' first.")
        except Exception:
            raise RuntimeError(
                f"Collection '{config.COLLECTION_NAME}' not found in {config.VECTORSTORE_DIR}. "
                f"Run 'python ingest.py' first to create the index."
            )

        # Load BM25 index if available
        self.bm25 = None
        self.bm25_chunk_ids = []
        if getattr(config, "USE_HYBRID_SEARCH", True):
            self._load_bm25_index()

        self.last_results = None

        # Load Cross-Encoder for re-ranking if enabled
        self.reranker = None
        if getattr(config, "USE_RERANKER", False):
            try:
                from sentence_transformers import CrossEncoder

                self.logger.info(
                    f"Loading re-ranker model: {config.RERANKER_MODEL} on [{self.device.upper()}]..."
                )
                self.reranker = CrossEncoder(config.RERANKER_MODEL, device=self.device)
            except Exception as e:
                self.logger.error(
                    f"Failed to load re-ranker model: {e}. Falling back to RRF rankings."
                )

    def _load_bm25_index(self):
        """Load the pre-computed BM25 index from disk."""
        if config.BM25_INDEX_PATH.exists():
            try:
                with open(config.BM25_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                corpus_tokens = data.get("corpus_tokens", [])
                self.bm25_chunk_ids = data.get("chunk_ids", [])
                if corpus_tokens:
                    self.bm25 = BM25Okapi(corpus_tokens)
                    self.logger.info(
                        f"BM25 index loaded ({len(self.bm25_chunk_ids)} documents)"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load BM25 index: {e}. Running vector-only retrieval."
                )
        else:
            self.logger.info(
                "BM25 index file not found. Running vector-only retrieval."
            )

    def _vector_search(self, query: str, fetch_k: int) -> list[dict]:
        """Perform dense semantic search in ChromaDB."""
        query_text = f"{config.QUERY_PREFIX}{query}"
        query_embedding = self.embedder.encode(
            [query_text], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            ids = results.get("ids", [[]])[0]
            for i, doc in enumerate(results["documents"][0]):
                doc_id = ids[i] if i < len(ids) else ""
                dist = results["distances"][0][i] if results.get("distances") else 0.0
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                retrieved.append(
                    {
                        "id": doc_id,
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "vector_similarity": 1.0 - dist,
                    }
                )
        return retrieved

    def _bm25_search(self, query: str, fetch_k: int) -> list[dict]:
        """Perform sparse keyword search using BM25."""
        if not self.bm25 or not self.bm25_chunk_ids:
            return []

        tokens = code_aware_tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:fetch_k]

        candidate_ids = [
            self.bm25_chunk_ids[i] for i in top_indices if scores[i] > 0
        ]
        if not candidate_ids:
            return []

        # Fetch document contents from ChromaDB by IDs
        docs_data = self.collection.get(
            ids=candidate_ids, include=["documents", "metadatas"]
        )

        id_to_doc = {}
        if docs_data and docs_data["ids"]:
            for i, d_id in enumerate(docs_data["ids"]):
                id_to_doc[d_id] = {
                    "text": docs_data["documents"][i] if docs_data.get("documents") else "",
                    "metadata": docs_data["metadatas"][i] if docs_data.get("metadatas") else {},
                }

        retrieved = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc_id = self.bm25_chunk_ids[idx]
            if doc_id in id_to_doc:
                info = id_to_doc[doc_id]
                retrieved.append(
                    {
                        "id": doc_id,
                        "text": info["text"],
                        "metadata": info["metadata"],
                        "bm25_score": float(score),
                    }
                )
        return retrieved

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        rrf_k: int = config.RRF_K,
        vector_weight: float = config.VECTOR_WEIGHT,
        bm25_weight: float = config.BM25_WEIGHT,
    ) -> list[dict]:
        """
        Merge vector and BM25 ranked candidate lists using weighted Reciprocal Rank Fusion (RRF).
        Formula: RRF_score(d) = vector_weight / (rrf_k + rank_v) + bm25_weight / (rrf_k + rank_b)
        """
        doc_map = {}
        rrf_scores = {}

        # Process vector ranks
        for rank, item in enumerate(vector_results):
            doc_id = item.get("id") or item["text"][:64]
            if doc_id not in doc_map:
                doc_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                vector_weight / (rrf_k + rank + 1)
            )

        # Process BM25 ranks
        for rank, item in enumerate(bm25_results):
            doc_id = item.get("id") or item["text"][:64]
            if doc_id not in doc_map:
                doc_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                bm25_weight / (rrf_k + rank + 1)
            )

        # Build merged result list
        merged = []
        for doc_id, score in rrf_scores.items():
            entry = dict(doc_map[doc_id])
            entry["rrf_score"] = score
            merged.append(entry)

        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged

    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """
        Execute hybrid search + re-ranking pipeline.
        Returns top-k most relevant document chunks.
        """
        fetch_k = getattr(config, "RETRIEVAL_CANDIDATES", 30)

        # Step 1: Dense vector search
        vector_results = self._vector_search(query, fetch_k=fetch_k)

        # Step 2: Sparse BM25 search
        bm25_results = (
            self._bm25_search(query, fetch_k=fetch_k)
            if self.bm25
            else []
        )

        # Step 3: Reciprocal Rank Fusion
        if bm25_results:
            candidates = self._reciprocal_rank_fusion(
                vector_results, bm25_results
            )[:fetch_k]
        else:
            candidates = vector_results

        if not candidates:
            self.last_results = []
            return []

        # Step 4: Cross-Encoder Re-ranking
        if self.reranker and candidates:
            try:
                pairs = [(query, c["text"]) for c in candidates]
                scores = self.reranker.predict(pairs)
                for idx, score in enumerate(scores):
                    candidates[idx]["rerank_score"] = float(score)
                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                final_results = candidates[:top_k]
            except Exception as e:
                self.logger.error(f"Error during re-ranking: {e}. Using RRF/Vector scores.")
                final_results = candidates[:top_k]
        else:
            final_results = candidates[:top_k]

        self.last_results = final_results
        return final_results

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks into a context string for prompt injection."""
        if not results:
            return "(No relevant documents found)"

        # Merge contiguous chunks for seamless context flow
        merged_results = self.merge_and_group_chunks(results)

        parts = []
        for i, r in enumerate(merged_results, 1):
            meta = r.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            file_path = meta.get("file_path", source)
            content_type = meta.get("content_type", "prose")

            if "rerank_score" in r:
                score_str = f"Relevance Score: {r['rerank_score']:.4f}"
            elif "rrf_score" in r:
                score_str = f"RRF Score: {r['rrf_score']:.4f}"
            elif "distance" in r:
                sim = 1.0 - r["distance"]
                score_str = f"Relevance: {sim:.2%}"
            else:
                score_str = "Relevance: High"

            parts.append(
                f"--- Chunk {i} [File: {file_path}, Source: {source}, Page: {page}, Type: {content_type}] "
                f"({score_str}) ---\n{r['text']}"
            )
        return "\n\n".join(parts)

    def merge_and_group_chunks(self, results: list[dict]) -> list[dict]:
        """Merge adjacent text chunks from the same page and document."""
        if not results:
            return []

        sorted_results = sorted(
            results,
            key=lambda x: (
                x.get("metadata", {}).get("file_path", ""),
                x.get("metadata", {}).get("page", 0),
                x.get("metadata", {}).get("chunk_index", 0),
            ),
        )

        merged = []
        current = None

        for r in sorted_results:
            if current is None:
                current = dict(r)
                continue

            curr_meta = current.get("metadata", {})
            r_meta = r.get("metadata", {})

            is_same_file = curr_meta.get("file_path") == r_meta.get("file_path")
            is_same_page = curr_meta.get("page") == r_meta.get("page")

            curr_idx = curr_meta.get("chunk_index")
            r_idx = r_meta.get("chunk_index")
            is_consecutive = (
                curr_idx is not None and r_idx is not None and curr_idx + 1 == r_idx
            )

            if is_same_file and is_same_page and is_consecutive:
                separator = "\n" if current["text"].endswith(("\n", "\r")) else " "
                current["text"] += separator + r["text"]
                curr_meta["chunk_index"] = r_idx
                curr_meta["char_count"] = len(current["text"])
                if "distance" in current and "distance" in r:
                    current["distance"] = min(current["distance"], r["distance"])
                if "rerank_score" in current and "rerank_score" in r:
                    current["rerank_score"] = max(
                        current["rerank_score"], r["rerank_score"]
                    )
            else:
                merged.append(current)
                current = dict(r)

        if current:
            merged.append(current)

        return merged

    def count(self) -> int:
        """Return total document count in the vector store."""
        return self.collection.count()


# ──────────────────────────────────────────────
# Ollama LLM Client (Qwen3:8B)
# ──────────────────────────────────────────────


class OllamaClient:
    """
    HTTP client for the Ollama local LLM API (Qwen3:8B).
    Supports streaming generation for responsive CLI output.
    """

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
        timeout: int = config.OLLAMA_TIMEOUT,
    ):
        self.logger = logging.getLogger("pdf_chatbot_v2.ollama")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

            model_found = any(
                self.model in m or m.startswith(self.model.split(":")[0])
                for m in models
            )

            if model_found:
                self.logger.info(f"✅ Ollama connected — model '{self.model}' available")
            elif models:
                fallback_model = models[0]
                self.logger.warning(
                    f"⚠ Configured model '{self.model}' not found in Ollama. "
                    f"Falling back to installed model '{fallback_model}'. "
                    f"To use Qwen3, run: ollama pull {self.model}"
                )
                self.model = fallback_model
            else:
                self.logger.warning(
                    f"⚠ Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. "
                    f"Run: ollama pull {self.model}"
                )
        except requests.ConnectionError:
            self.logger.error(
                f"✖ Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: 'ollama serve'"
            )
            raise SystemExit(1)
        except Exception as e:
            self.logger.warning(f"⚠ Ollama check warning: {e}")

    def generate(
        self, prompt: str, system: str = SYSTEM_PROMPT, stream: bool = True
    ) -> str:
        """
        Send a prompt to Ollama and stream or return the generated response.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": stream,
            "options": {
                "num_ctx": config.OLLAMA_CONTEXT_WINDOW,  # 8192 context window (fits 100% in VRAM)
                "num_predict": 1536,                      # Bounded generation prevents runaway loops
                "temperature": 0.3,                       # Accurate factual grounding
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=self.timeout,
            )
            response.raise_for_status()

            if stream:
                full_response = []
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        full_response.append(token)
                        print(token, end="", flush=True)
                        if data.get("done", False):
                            break
                print()  # Newline after streaming
                return "".join(full_response).strip()
            else:
                data = response.json()
                return data.get("response", "").strip()

        except requests.Timeout:
            self.logger.error(f"Ollama timed out after {self.timeout}s.")
            return "⚠ Response timed out. Please try a shorter question or check system load."
        except requests.ConnectionError:
            self.logger.error("Lost connection to Ollama during generation.")
            return "⚠ Lost connection to Ollama. Is it still running?"
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return f"⚠ Error: {e}"


# ──────────────────────────────────────────────
# Chat Engine
# ──────────────────────────────────────────────


class ChatEngine:
    """
    Orchestrates the full PDF-Chatbot-V2 RAG pipeline:
      Query → Embed → Hybrid Search (BM25+Vector+RRF) → Re-ranking → Prompt → Qwen3 Generate → Memory
    """

    def __init__(self, top_k: int = config.TOP_K):
        self.logger = logging.getLogger("pdf_chatbot_v2.chat")
        self.console = Console()
        self.top_k = top_k

        self.console.print(
            Panel(
                "[bold cyan]PDF-Chatbot-V2[/bold cyan] — Lightweight Hybrid RAG Pipeline\n"
                "Initializing Hybrid Retriever + Qwen3 Engine...",
                border_style="cyan",
            )
        )

        self.retriever = Retriever()
        self.llm = OllamaClient()
        self.memory = ConversationMemory()

        hybrid_status = "Enabled (BM25 + Vector + RRF)" if self.retriever.bm25 else "Vector Only"
        profile = config.get_active_profile()
        self.console.print(
            Panel(
                f"[bold green]✅ Ready![/bold green]\n"
                f"⚡ Mode: [bold cyan]{profile['mode_name']}[/bold cyan]\n"
                f"📄 [bold]{self.retriever.count()}[/bold] document chunks indexed\n"
                f"🤖 Model: [bold]{config.OLLAMA_MODEL}[/bold] (Context: {config.OLLAMA_CONTEXT_WINDOW} tokens)\n"
                f"🔍 Retrieval: [bold]{hybrid_status}[/bold] (Candidates: {config.RETRIEVAL_CANDIDATES})\n"
                f"🎯 Top-K: [bold]{self.top_k}[/bold] | Timeout: {config.OLLAMA_TIMEOUT}s\n\n"
                f"Type your question or /help for commands.",
                title="[bold]System Status[/bold]",
                border_style="green",
            )
        )

    def ask(self, question: str) -> str:
        """Process a single question through the full hybrid RAG pipeline."""
        self.logger.info(f"Query: {question}")
        start = time.time()

        # Step 1: Hybrid retrieval + Re-ranking
        results = self.retriever.search(question, top_k=self.top_k)
        context = self.retriever.format_context(results)

        if not results:
            self.console.print("[yellow]⚠ No relevant documents found.[/yellow]")

        # Step 2: Build prompt with conversation history
        history = self.memory.format_history()
        prompt = QUERY_TEMPLATE.format(
            context=context,
            history=history,
            question=question,
        )

        # Step 3: Generate answer
        self.console.print("\n[bold cyan]Assistant:[/bold cyan]")
        answer = self.llm.generate(prompt)

        # Step 4: Update memory
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", answer)

        # Step 5: Show source summary
        elapsed = time.time() - start
        self._show_sources_brief(results, elapsed)

        return answer

    def _show_sources_brief(self, results: list[dict], elapsed: float):
        """Display a compact source summary line below the answer."""
        if not results:
            return

        sources = set()
        for r in results:
            meta = r.get("metadata", {})
            sources.add(f"{meta.get('source', '?')}:p{meta.get('page', '?')}")

        source_str = ", ".join(sorted(sources)[:5])
        self.console.print(
            f"\n[dim]📎 Sources: {source_str} | "
            f"⏱ {elapsed:.1f}s | "
            f"💬 Memory: {self.memory.turn_count} turns[/dim]"
        )

    def show_sources_detail(self):
        """Show a detailed breakdown table of the last retrieved sources."""
        results = self.retriever.last_results
        if not results:
            self.console.print("[yellow]No previous retrieval to show.[/yellow]")
            return

        table = Table(title="Last Retrieval Sources (PDF-Chatbot-V2)", border_style="cyan")
        table.add_column("#", justify="center", width=3)
        table.add_column("Source", style="green")
        table.add_column("Page", justify="center")
        table.add_column("Type", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Preview", max_width=50)

        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            ctype = meta.get("content_type", "prose")

            if "rerank_score" in r:
                score_str = f"Rerank: {r['rerank_score']:.3f}"
            elif "rrf_score" in r:
                score_str = f"RRF: {r['rrf_score']:.4f}"
            elif "distance" in r:
                sim = 1.0 - r["distance"]
                score_str = f"{sim:.1%}"
            else:
                score_str = "N/A"

            preview = r["text"][:80].replace("\n", " ") + "..."
            table.add_row(
                str(i),
                meta.get("source", "?"),
                str(meta.get("page", "?")),
                ctype,
                score_str,
                preview,
            )

        self.console.print(table)

    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if the chat should terminate."""
        cmd = command.strip().lower()

        if cmd in ("/quit", "/exit"):
            self.console.print("[bold red]Goodbye! 👋[/bold red]")
            return True

        elif cmd == "/clear":
            self.memory.clear()
            self.console.print("[green]🔄 Conversation memory cleared.[/green]")

        elif cmd == "/sources":
            self.show_sources_detail()

        elif cmd == "/count":
            count = self.retriever.count()
            self.console.print(f"[cyan]📊 {count} document chunks in vector store[/cyan]")

        elif cmd == "/help":
            help_table = Table(title="Available Commands", border_style="cyan")
            help_table.add_column("Command", style="bold green")
            help_table.add_column("Description")
            help_table.add_row("/clear", "Reset conversation memory")
            help_table.add_row("/sources", "Show detailed sources from last query")
            help_table.add_row("/count", "Show total indexed document chunks")
            help_table.add_row("/help", "Show this help message")
            help_table.add_row("/quit", "Exit the chatbot")
            self.console.print(help_table)

        else:
            self.console.print(f"[yellow]Unknown command: {cmd}. Type /help[/yellow]")

        return False

    def run_interactive(self):
        """Run the interactive CLI loop."""
        while True:
            try:
                self.console.print()
                question = input("📝 You: ").strip()

                if not question:
                    continue

                if question.startswith("/"):
                    if self.handle_command(question):
                        break
                    continue

                self.ask(question)

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Interrupted. Goodbye! 👋[/bold red]")
                break
            except EOFError:
                break

    def run_single(self, question: str):
        """Run a single query and exit."""
        self.ask(question)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF-Chatbot-V2 — Hybrid Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python query.py                          # Interactive mode\n"
            '  python query.py --query "Summarize X"    # Single-shot query\n'
            "  python query.py --top-k 8                # Context chunks\n"
        ),
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Single-shot query (non-interactive mode)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of context chunks to retrieve (default: {config.TOP_K})",
    )
    args = parser.parse_args()

    setup_logging()
    chat = ChatEngine(top_k=args.top_k)

    if args.query:
        chat.run_single(args.query)
    else:
        chat.run_interactive()
