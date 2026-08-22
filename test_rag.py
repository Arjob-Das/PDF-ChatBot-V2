"""
PDF-Chatbot-V2 — Validation & Testing Suite (Phase 4)
=====================================================
Validates the hybrid RAG pipeline with 6 comprehensive test suites:
  1. Index Integrity Check (ChromaDB collection, documents, metadata, IDs)
  2. BM25 Index Integrity Check (vocabulary, document mapping, scores)
  3. Embedding Model Benchmark (nomic-embed speed, 768-dim, normalization)
  4. Hybrid Retrieval Quality (cosine similarity & ranking on diverse queries)
  5. QA Evaluation Loop (faithfulness & answer relevancy scoring)
  6. Ingestion Speed Benchmark (structure-aware chunking & embedding throughput)

Usage:
    python test_rag.py                  # Run all tests
    python test_rag.py --skip-llm       # Skip LLM-based tests (no Ollama needed)
    python test_rag.py --verbose        # Detailed output
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

import config
from query import Retriever
from utils import detect_device, format_time, setup_logging

console = Console()

# ──────────────────────────────────────────────
# Ground-Truth QA Pairs (Mixed Content)
# ──────────────────────────────────────────────
# Covers general overview, procedural instructions,
# technical specifications, code concepts, and recommendations.

GROUND_TRUTH_QA = [
    {
        "question": "What are the main topics and objectives covered in the documents?",
        "expected_keywords": ["document", "topic", "overview", "system", "architecture"],
        "description": "General content overview",
        "domain": "general",
    },
    {
        "question": "What are the step-by-step instructions or procedures outlined?",
        "expected_keywords": ["step", "procedure", "configure", "install", "run"],
        "description": "Procedural instructions",
        "domain": "instructional",
    },
    {
        "question": "What technical specifications, APIs, or architectural components are defined?",
        "expected_keywords": ["api", "module", "component", "service", "interface"],
        "description": "Technical specifications",
        "domain": "technical",
    },
    {
        "question": "Are there code examples, functions, or shell commands provided?",
        "expected_keywords": ["function", "code", "command", "script", "import"],
        "description": "Code & CLI extraction",
        "domain": "code",
    },
    {
        "question": "What are the key conclusions, findings, or recommendations?",
        "expected_keywords": ["recommend", "conclusion", "result", "finding", "summary"],
        "description": "Key takeaways & recommendations",
        "domain": "analytical",
    },
]


# ──────────────────────────────────────────────
# Test 1: Index Integrity (ChromaDB)
# ──────────────────────────────────────────────


def test_index_integrity(verbose: bool = False) -> dict:
    """Verify ChromaDB vector store collection exists and has valid records."""
    console.print("\n[bold]🔍 Test 1: ChromaDB Index Integrity Check[/bold]")
    result = {"passed": False, "details": "", "doc_count": 0}

    try:
        if not config.VECTORSTORE_DIR.exists():
            result["details"] = (
                f"Vector store directory not found: {config.VECTORSTORE_DIR}. "
                f"Run 'python ingest.py' first."
            )
            console.print(f"  [red]✖ FAIL: {result['details']}[/red]")
            return result

        client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))

        try:
            collection = client.get_collection(name=config.COLLECTION_NAME)
        except Exception:
            result["details"] = (
                f"Collection '{config.COLLECTION_NAME}' not found. "
                f"Run 'python ingest.py' first."
            )
            console.print(f"  [red]✖ FAIL: {result['details']}[/red]")
            return result

        doc_count = collection.count()
        result["doc_count"] = doc_count

        if doc_count == 0:
            result["details"] = "Collection exists but is empty."
            console.print(f"  [yellow]⚠ WARN: {result['details']}[/yellow]")
            return result

        sample = collection.peek(limit=1)
        has_docs = bool(sample.get("documents"))
        has_meta = bool(sample.get("metadatas"))
        has_ids = bool(sample.get("ids"))

        if verbose and has_docs:
            console.print(f"  Sample doc preview: {sample['documents'][0][:100]}...")
            console.print(f"  Sample metadata: {sample['metadatas'][0]}")

        result["passed"] = has_docs and has_meta and has_ids
        result["details"] = (
            f"{doc_count} docs | "
            f"docs={'✅' if has_docs else '❌'} "
            f"meta={'✅' if has_meta else '❌'} "
            f"ids={'✅' if has_ids else '❌'}"
        )

        status = "[green]✅ PASS" if result["passed"] else "[red]✖ FAIL"
        console.print(f"  {status}: {result['details']}[/]")

    except Exception as e:
        result["details"] = f"Error: {e}"
        console.print(f"  [red]✖ FAIL: {result['details']}[/red]")

    return result


# ──────────────────────────────────────────────
# Test 2: BM25 Index Integrity
# ──────────────────────────────────────────────


def test_bm25_integrity(verbose: bool = False) -> dict:
    """Verify BM25 sparse index exists, is valid, and matches document records."""
    console.print("\n[bold]📚 Test 2: BM25 Sparse Index Integrity Check[/bold]")
    result = {"passed": False, "details": "", "doc_count": 0}

    try:
        if not config.BM25_INDEX_PATH.exists():
            result["details"] = (
                f"BM25 index not found at '{config.BM25_INDEX_PATH}'. "
                f"Run 'python ingest.py' to generate hybrid indexes."
            )
            console.print(f"  [yellow]⚠ WARN: {result['details']}[/yellow]")
            return result

        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)

        corpus_tokens = data.get("corpus_tokens", [])
        chunk_ids = data.get("chunk_ids", [])

        doc_count = len(chunk_ids)
        result["doc_count"] = doc_count
        has_tokens = len(corpus_tokens) > 0
        has_ids = len(chunk_ids) > 0
        counts_match = len(corpus_tokens) == len(chunk_ids)

        if has_tokens:
            total_tokens = sum(len(doc) for doc in corpus_tokens)
            avg_tokens = total_tokens / max(len(corpus_tokens), 1)
        else:
            avg_tokens = 0

        result["passed"] = has_tokens and has_ids and counts_match
        result["details"] = (
            f"{doc_count} documents | "
            f"avg {avg_tokens:.0f} tokens/doc | "
            f"sync={'✅' if counts_match else '❌'}"
        )

        status = "[green]✅ PASS" if result["passed"] else "[red]✖ FAIL"
        console.print(f"  {status}: {result['details']}[/]")

        if verbose and has_tokens:
            console.print(f"  Sample tokenized doc: {corpus_tokens[0][:10]}...")

    except Exception as e:
        result["details"] = f"Error: {e}"
        console.print(f"  [red]✖ FAIL: {result['details']}[/red]")

    return result


# ──────────────────────────────────────────────
# Test 3: Embedding Model Benchmark
# ──────────────────────────────────────────────


def test_embedding_benchmark(verbose: bool = False) -> dict:
    """Benchmark nomic-embed embedding throughput and verify 768-dim output."""
    console.print("\n[bold]⚡ Test 3: Embedding Model Benchmark (nomic-embed-text-v1.5)[/bold]")
    result = {"passed": False, "details": "", "speed": 0.0}

    try:
        device = detect_device()
        console.print(f"  Device: [{device.upper()}]")

        model = SentenceTransformer(
            config.EMBEDDING_MODEL, device=device, trust_remote_code=True
        )

        sample_texts = [
            f"{config.EMBEDDING_PREFIX}The quick brown fox jumps over the lazy dog.",
            f"{config.EMBEDDING_PREFIX}def calculate_metrics(data: list) -> dict: return {{'mean': sum(data)/len(data)}}",
            f"{config.EMBEDDING_PREFIX}Docker container orchestration using Kubernetes pods and deployments.",
            f"{config.EMBEDDING_PREFIX}REST API design principles: idempotency, resource-oriented URIs, status codes.",
            f"{config.EMBEDDING_PREFIX}Instructional manual for configuring enterprise database connection pools.",
        ] * 10  # 50 texts

        start = time.time()
        embeddings = model.encode(sample_texts, normalize_embeddings=True)
        elapsed = time.time() - start

        dim = embeddings.shape[1]
        expected_dim = config.EMBEDDING_DIMENSION  # 768
        dim_ok = dim == expected_dim
        shape_ok = embeddings.shape[0] == len(sample_texts)
        norm_ok = all(abs(np.linalg.norm(e) - 1.0) < 0.01 for e in embeddings[:5])

        speed = len(sample_texts) / max(elapsed, 0.001)
        result["speed"] = speed
        result["passed"] = dim_ok and shape_ok and norm_ok

        result["details"] = (
            f"{len(sample_texts)} texts in {format_time(elapsed)} "
            f"({speed:.0f} texts/sec) | "
            f"dim={dim}{'✅' if dim_ok else '❌'} (expected {expected_dim}) | "
            f"normalized={'✅' if norm_ok else '❌'}"
        )

        status = "[green]✅ PASS" if result["passed"] else "[red]✖ FAIL"
        console.print(f"  {status}: {result['details']}[/]")

        if verbose:
            console.print(f"  Output shape: {embeddings.shape}")
            console.print(f"  L2 norm (sample): {np.linalg.norm(embeddings[0]):.4f}")

    except Exception as e:
        result["details"] = f"Error: {e}"
        console.print(f"  [red]✖ FAIL: {result['details']}[/red]")

    return result


# ──────────────────────────────────────────────
# Test 4: Retrieval Quality
# ──────────────────────────────────────────────


def test_retrieval_quality(verbose: bool = False) -> dict:
    """Test hybrid search retrieval quality across diverse queries."""
    console.print("\n[bold]🎯 Test 4: Hybrid Retrieval Quality Check[/bold]")
    result = {"passed": False, "details": "", "avg_relevancy": 0.0}

    try:
        retriever = Retriever()
        relevancy_scores = []

        for qa in GROUND_TRUTH_QA:
            results = retriever.search(qa["question"])

            if results:
                similarities = []
                for r in results:
                    if "rerank_score" in r:
                        # Convert logit or cross-encoder score to 0-1 range
                        score = 1.0 / (1.0 + np.exp(-r["rerank_score"]))
                        similarities.append(float(score))
                    elif "distance" in r:
                        similarities.append(float(1.0 - r["distance"]))
                    elif "rrf_score" in r:
                        similarities.append(min(r["rrf_score"] * 30.0, 1.0))

                if similarities:
                    avg_sim = np.mean(similarities)
                    relevancy_scores.append(avg_sim)

                    if verbose:
                        console.print(
                            f"  Q [{qa['domain']}]: {qa['question'][:50]}... → "
                            f"score={avg_sim:.3f}"
                        )

        if relevancy_scores:
            avg_relevancy = np.mean(relevancy_scores)
            result["avg_relevancy"] = float(avg_relevancy)
            result["passed"] = avg_relevancy > 0.20
            result["details"] = (
                f"Avg relevancy: {avg_relevancy:.3f} across "
                f"{len(relevancy_scores)} queries"
            )
        else:
            result["details"] = "No retrieval results returned (empty store)"

        status = "[green]✅ PASS" if result["passed"] else "[red]✖ FAIL"
        console.print(f"  {status}: {result['details']}[/]")

    except Exception as e:
        result["details"] = f"Error: {e}"
        console.print(f"  [red]✖ FAIL: {result['details']}[/red]")

    return result


# ──────────────────────────────────────────────
# Test 5: QA Evaluation Loop
# ──────────────────────────────────────────────


def test_qa_evaluation(skip_llm: bool = False, verbose: bool = False) -> dict:
    """Evaluate pipeline on ground-truth questions with Faithfulness and Relevancy scoring."""
    console.print("\n[bold]📝 Test 5: QA Evaluation Loop[/bold]")
    result = {"passed": False, "details": "", "scores": []}

    try:
        retriever = Retriever()

        ollama_available = False
        if not skip_llm:
            try:
                resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
                ollama_available = resp.status_code == 200
            except Exception:
                console.print(
                    "  [yellow]⚠ Ollama not available — evaluating via deterministic metrics[/yellow]"
                )

        scores_table = Table(title="QA Evaluation Results (PDF-Chatbot-V2)", border_style="cyan")
        scores_table.add_column("#", justify="center", width=3)
        scores_table.add_column("Domain", style="magenta", width=12)
        scores_table.add_column("Question", max_width=35)
        scores_table.add_column("Faithfulness", justify="center")
        scores_table.add_column("Relevancy", justify="center")
        if ollama_available:
            scores_table.add_column("LLM Judge", justify="center")

        all_faithfulness = []
        all_relevancy = []
        all_llm_scores = []

        for i, qa in enumerate(GROUND_TRUTH_QA, 1):
            results = retriever.search(qa["question"])

            retrieved_text = ""
            avg_similarity = 0.0
            if results:
                retrieved_text = retriever.format_context(results)
                sims = [1.0 - r.get("distance", 0.5) for r in results if "distance" in r]
                avg_similarity = np.mean(sims) if sims else 0.5

            # Deterministic keyword match (Faithfulness proxy)
            if retrieved_text and qa["expected_keywords"]:
                matched = sum(
                    1
                    for kw in qa["expected_keywords"]
                    if kw.lower() in retrieved_text.lower()
                )
                faithfulness = matched / len(qa["expected_keywords"])
            else:
                faithfulness = 0.0

            all_faithfulness.append(faithfulness)
            all_relevancy.append(float(avg_similarity))

            llm_score = None
            if ollama_available and retrieved_text:
                llm_score = _llm_judge_score(qa["question"], retrieved_text)
                if llm_score is not None:
                    all_llm_scores.append(llm_score)

            row = [
                str(i),
                qa["domain"],
                qa["question"][:35],
                f"{faithfulness:.0%}",
                f"{avg_similarity:.3f}",
            ]
            if ollama_available:
                row.append(f"{llm_score}/5" if llm_score else "N/A")

            scores_table.add_row(*row)

            result["scores"].append(
                {
                    "question": qa["question"],
                    "domain": qa["domain"],
                    "faithfulness": faithfulness,
                    "relevancy": float(avg_similarity),
                    "llm_score": llm_score,
                }
            )

        console.print(scores_table)

        avg_faith = np.mean(all_faithfulness) if all_faithfulness else 0.0
        avg_rel = np.mean(all_relevancy) if all_relevancy else 0.0
        avg_llm = np.mean(all_llm_scores) if all_llm_scores else None

        console.print("\n  [bold]Aggregated Metrics:[/bold]")
        console.print(f"    Avg Faithfulness : {avg_faith:.1%}")
        console.print(f"    Avg Relevancy    : {avg_rel:.3f}")
        if avg_llm is not None:
            console.print(f"    Avg LLM Judge    : {avg_llm:.1f}/5")

        result["passed"] = avg_rel > 0.20 or avg_faith > 0.10
        result["details"] = (
            f"Faith={avg_faith:.1%}, Rel={avg_rel:.3f}"
            + (f", LLM={avg_llm:.1f}/5" if avg_llm else "")
        )

        status = "[green]✅ PASS" if result["passed"] else "[red]✖ FAIL"
        console.print(f"\n  {status}: {result['details']}[/]")

    except Exception as e:
        result["details"] = f"Error: {e}"
        console.print(f"  [red]✖ FAIL: {result['details']}[/red]")

    return result


def _llm_judge_score(question: str, context: str) -> int | None:
    """Use Ollama Qwen3:8B to judge retrieved context relevance (1-5 scale)."""
    judge_prompt = f"""Rate how well the following retrieved context can answer the question.
Score from 1 to 5:
  1 = Completely irrelevant
  2 = Slightly related but insufficient
  3 = Partially relevant, some useful information
  4 = Mostly relevant, good answer possible
  5 = Highly relevant, excellent answer possible

Question: {question}

Retrieved Context (sample):
{context[:600]}

Respond with ONLY a single digit integer (1 to 5), nothing else."""

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": judge_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 2048},
            },
            timeout=20,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()

        for char in answer:
            if char.isdigit() and 1 <= int(char) <= 5:
                return int(char)
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────
# Test 6: Ingestion Speed Benchmark
# ──────────────────────────────────────────────


def test_ingestion_benchmark(verbose: bool = False) -> dict:
    """Benchmark structure-aware chunking and nomic-embed throughput."""
    console.print("\n[bold]🚀 Test 6: Ingestion Speed Benchmark[/bold]")
    result = {"passed": False, "details": "", "chunks_per_sec": 0.0}

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        sample_prose = (
            "This section covers system configuration and operational procedures. "
            "Administrators should ensure environmental variables are properly loaded. "
            "The architecture employs microservices communicating via REST and gRPC. "
        ) * 20

        sample_code = (
            "\n```python\n"
            "def deploy_service(config: dict) -> bool:\n"
            "    # Deploy container to Kubernetes cluster\n"
            "    client = KubernetesClient(config['cluster'])\n"
            "    return client.apply_manifest(config['manifest'])\n"
            "```\n"
        ) * 10

        full_sample = (sample_prose + sample_code) * 5

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=[
                "\n\n\n", "\n\n", "\ndef ", "\nclass ", "\n```", "\n", ". ", " ", ""
            ],
        )

        start = time.time()
        chunks = splitter.split_text(full_sample)
        chunk_time = time.time() - start

        device = detect_device()
        model = SentenceTransformer(
            config.EMBEDDING_MODEL, device=device, trust_remote_code=True
        )

        texts = [f"{config.EMBEDDING_PREFIX}{c}" for c in chunks[:80]]

        start = time.time()
        model.encode(texts, normalize_embeddings=True, batch_size=32)
        embed_time = time.time() - start

        chunks_per_sec = len(texts) / max(embed_time, 0.001)
        result["chunks_per_sec"] = chunks_per_sec
        result["passed"] = chunks_per_sec > 1.0

        result["details"] = (
            f"Chunking: {len(chunks)} chunks in {format_time(chunk_time)} | "
            f"Embedding: {len(texts)} chunks in {format_time(embed_time)} "
            f"({chunks_per_sec:.0f} chunks/sec)"
        )

        status = "[green]✅ PASS" if result["passed"] else "[red]✖ FAIL"
        console.print(f"  {status}: {result['details']}[/]")

        if verbose:
            console.print(f"  Device: {device}")
            console.print(f"  Chunks generated: {len(chunks)}")

    except Exception as e:
        result["details"] = f"Error: {e}"
        console.print(f"  [red]✖ FAIL: {result['details']}[/red]")

    return result


# ──────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────


def run_all_tests(skip_llm: bool = False, verbose: bool = False) -> dict:
    """Execute all test suites and output structured summary report."""
    console.print(
        Panel(
            "[bold cyan]PDF-Chatbot-V2 — Validation & Testing Suite[/bold cyan]\n"
            "Running comprehensive health and performance checks...",
            border_style="cyan",
        )
    )

    start = time.time()
    results = {
        "index_integrity": test_index_integrity(verbose),
        "bm25_integrity": test_bm25_integrity(verbose),
        "embedding_benchmark": test_embedding_benchmark(verbose),
        "retrieval_quality": test_retrieval_quality(verbose),
        "qa_evaluation": test_qa_evaluation(skip_llm, verbose),
        "ingestion_benchmark": test_ingestion_benchmark(verbose),
    }

    elapsed = time.time() - start
    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)

    console.print("\n" + "=" * 60)
    summary_table = Table(title="Test Summary (PDF-Chatbot-V2)", border_style="cyan")
    summary_table.add_column("Test Suite", style="bold")
    summary_table.add_column("Result", justify="center")
    summary_table.add_column("Details", max_width=50)

    for name, res in results.items():
        status = "✅ PASS" if res["passed"] else "❌ FAIL"
        style = "green" if res["passed"] else "red"
        summary_table.add_row(
            name.replace("_", " ").title(),
            f"[{style}]{status}[/{style}]",
            res["details"][:50],
        )

    console.print(summary_table)
    console.print(f"\n  [bold]{passed}/{total} tests passed[/bold] in {format_time(elapsed)}")

    if passed == total:
        console.print("  [bold green]🎉 All PDF-Chatbot-V2 tests passed![/bold green]")
    else:
        console.print(f"  [bold yellow]⚠ {total - passed} test(s) need attention[/bold yellow]")

    return results


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF-Chatbot-V2 — Validation & Testing Suite",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based evaluation (no Ollama required)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for each test",
    )
    args = parser.parse_args()

    setup_logging()
    results = run_all_tests(skip_llm=args.skip_llm, verbose=args.verbose)

    all_passed = all(r["passed"] for r in results.values())
    sys.exit(0 if all_passed else 1)
