"""
PDF-Chatbot-V2 — Ingestion Engine
=================================
Batch-processes PDFs from a local directory:
  1. Extract text with PyMuPDF (fitz) + RapidOCR fallback
  2. Split into structure-aware chunks (512 chars, 64 overlap)
  3. Embed with nomic-embed-text-v1.5 (CUDA or CPU)
Processes PDF files and generates dual indexes:
  1. Extract text with PyMuPDF + RapidOCR fallback
  2. Recursive structure-aware chunking (512 chars, 64 overlap)
  3. Generate nomic-embed-text-v1.5 embeddings & store in ChromaDB
  4. Build & persist Code-Aware BM25 Okapi sparse index for hybrid search
"""

import argparse
import gc
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Generator

import chromadb
import numpy as np
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from utils import (
    code_aware_tokenize,
    detect_content_type,
    detect_device,
    format_time,
    hash_document_id,
    setup_logging,
)

# ──────────────────────────────────────────────
# PDF Discovery
# ──────────────────────────────────────────────


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """
    Find all PDF files in the given directory and all subdirectories (recursive).
    Returns a sorted list for deterministic processing order.
    """
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in '{pdf_dir}' (searched recursively). "
            f"Place your PDFs in this directory (or subfolders) and re-run."
        )
    return pdfs


# ──────────────────────────────────────────────
# Batch Generator (OOM Guard)
# ──────────────────────────────────────────────


def batch_generator(
    items: list, batch_size: int
) -> Generator[list, None, None]:
    """
    Yield successive batches of `batch_size` from `items`.
    This prevents loading all PDFs into memory simultaneously.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ──────────────────────────────────────────────
# Text Extraction (PyMuPDF) & OCR Support
# ──────────────────────────────────────────────

_rapidocr_engine = None


def _get_ocr_engine():
    """Lazily load RapidOCR engine (ONNX-based, lightweight)."""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            logging.getLogger("pdf_chatbot_v2.ocr").info(
                "Initializing RapidOCR engine (ONNX Runtime)..."
            )
            _rapidocr_engine = RapidOCR()
        except ImportError:
            logging.getLogger("pdf_chatbot_v2.ocr").error(
                "rapidocr-onnxruntime not installed. Install with: "
                "pip install rapidocr-onnxruntime"
            )
            raise
    return _rapidocr_engine


def extract_text_from_pdf(
    pdf_path: Path, logger: logging.Logger, enable_ocr: bool = config.ENABLE_OCR
) -> list[dict]:
    """
    Extract text from every page of a PDF using PyMuPDF.
    If the page is scanned (has no or very little text), falls back to RapidOCR.

    Returns:
        List of dicts: [{"text": str, "page": int, "source": str, "file_path": str}, ...]
        Empty list if the PDF is corrupt or has no extractable text.
    """
    pages = []
    try:
        doc = pymupdf.open(str(pdf_path))
        # Get relative path to project root for uniqueness
        try:
            rel_path = str(pdf_path.relative_to(config.PROJECT_ROOT))
        except ValueError:
            rel_path = str(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            # Check if page is empty or has minimal selectable text (scanned PDF page)
            if enable_ocr and len(text) < config.OCR_MIN_CHARACTERS:
                try:
                    engine = _get_ocr_engine()
                    # Render page directly to a NumPy array
                    pix = page.get_pixmap(dpi=config.OCR_DPI, colorspace=pymupdf.csRGB)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, 3
                    )

                    logger.info(
                        f"🔍 Running OCR on: {pdf_path.name} [Page {page_num + 1}]..."
                    )
                    # RapidOCR returns (result, elapsed_time)
                    # result is list of [box, text, confidence] or None
                    ocr_result, _ = engine(img_np)

                    if ocr_result:
                        ocr_text = " ".join(
                            item[1] for item in ocr_result
                        ).strip()

                        if len(ocr_text) > len(text):
                            text = ocr_text
                            logger.info(
                                f"   Success: extracted {len(text)} characters via OCR."
                            )
                except Exception as ocr_err:
                    logger.error(
                        f"   OCR failed on {pdf_path.name} "
                        f"[Page {page_num + 1}]: {ocr_err}"
                    )

            if text:
                pages.append(
                    {
                        "text": text,
                        "page": page_num + 1,  # 1-indexed for display
                        "source": pdf_path.name,
                        "file_path": rel_path,
                    }
                )
        doc.close()

        if not pages:
            logger.warning(f"⚠ No extractable text in: {pdf_path.name}")
    except Exception as e:
        logger.error(f"✖ Failed to read '{pdf_path.name}': {e}")

    return pages


# ──────────────────────────────────────────────
# Structure-Aware Text Chunking
# ──────────────────────────────────────────────

# Separators ordered by priority — structure-aware for both prose and code
_STRUCTURE_SEPARATORS = [
    "\n\n\n",           # Major section breaks
    "\n\n",             # Paragraph breaks
    "\n# ",             # Markdown headers
    "\n## ",
    "\n### ",
    "\ndef ",            # Python function definitions
    "\nclass ",          # Python/Java class definitions
    "\npublic ",         # Java access modifiers
    "\nprivate ",
    "\nprotected ",
    "\nfunction ",       # Shell/JS function definitions
    "\n```",             # Fenced code block boundaries
    "\n---",             # Horizontal rules / section dividers
    "\n",               # Line breaks
    ". ",               # Sentence boundaries
    " ",                # Word boundaries
    "",                 # Character-level fallback
]


def chunk_pages(
    pages: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split extracted page texts into overlapping chunks with structure awareness.

    Respects natural document boundaries:
      - Paragraph breaks and section headers (prose content)
      - Function/class definitions (code content)
      - Fenced code blocks and horizontal rules
      - Sentence boundaries as fallback

    Each chunk retains metadata: source filename, page number, chunk index,
    and content type (code/prose/mixed/table).

    Returns:
        List of dicts: [{"text": str, "id": str, "metadata": dict}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_STRUCTURE_SEPARATORS,
    )

    chunks = []
    for page_data in pages:
        splits = splitter.split_text(page_data["text"])
        for idx, split_text in enumerate(splits):
            doc_id = hash_document_id(
                page_data["file_path"], page_data["page"], idx
            )
            # Detect content type for metadata enrichment
            content_type = detect_content_type(split_text)

            chunks.append(
                {
                    "text": split_text,
                    "id": doc_id,
                    "metadata": {
                        "source": page_data["source"],
                        "file_path": page_data["file_path"],
                        "page": page_data["page"],
                        "chunk_index": idx,
                        "char_count": len(split_text),
                        "content_type": content_type,
                    },
                }
            )
    return chunks


# ──────────────────────────────────────────────
# Embedding Engine (nomic-embed-text-v1.5)
# ──────────────────────────────────────────────


class EmbeddingEngine:
    """
    Wraps SentenceTransformer for nomic-embed-text-v1.5 embedding generation.
    Auto-detects CUDA vs CPU. Encodes texts in sub-batches to
    control VRAM/RAM consumption.

    nomic-embed uses task prefixes for better accuracy:
      - "search_document: " for document embeddings (ingestion)
      - "search_query: " for query embeddings (retrieval)
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, device: str = None):
        self.logger = logging.getLogger("pdf_chatbot_v2.embedding")
        self.device = device or detect_device()

        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name, device=self.device, trust_remote_code=True
        )
        self.logger.info(
            f"Embedding model loaded on [{self.device.upper()}] — "
            f"dim={self.model.get_embedding_dimension()}"
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        prefix: str = config.EMBEDDING_PREFIX,
    ) -> list[list[float]]:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Sub-batch size for encoding (controls memory).
            prefix: Task prefix for nomic-embed ("search_document: " or "search_query: ").

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        # Prepend task prefix for nomic-embed
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )
        return embeddings.tolist()


# ──────────────────────────────────────────────
# ChromaDB Vector Store
# ──────────────────────────────────────────────


class VectorStore:
    """
    Manages ChromaDB collection for persistent vector storage.
    Supports upsert (deduplication via deterministic IDs) and reset.
    """

    def __init__(
        self,
        persist_dir: Path = config.VECTORSTORE_DIR,
        collection_name: str = config.COLLECTION_NAME,
    ):
        self.logger = logging.getLogger("pdf_chatbot_v2.vectorstore")
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initializing ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        self.logger.info(
            f"Collection '{collection_name}' ready — "
            f"{self.collection.count()} existing documents"
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
        chroma_max_batch_size: int = 5000,
    ) -> int:
        """
        Upsert documents into the collection.
        Deterministic IDs ensure re-running ingestion doesn't create duplicates.
        Chunks upserts to avoid exceeding ChromaDB maximum batch size (5461).

        Returns:
            Number of documents upserted.
        """
        total_upserted = 0
        for i in range(0, len(ids), chroma_max_batch_size):
            b_ids = ids[i : i + chroma_max_batch_size]
            b_embeddings = embeddings[i : i + chroma_max_batch_size]
            b_documents = documents[i : i + chroma_max_batch_size]
            b_metadatas = metadatas[i : i + chroma_max_batch_size]

            self.collection.upsert(
                ids=b_ids,
                embeddings=b_embeddings,
                documents=b_documents,
                metadatas=b_metadatas,
            )
            total_upserted += len(b_ids)
        return total_upserted

    def count(self) -> int:
        """Return total document count in the collection."""
        return self.collection.count()

    def reset(self, collection_name: str = config.COLLECTION_NAME):
        """Delete and recreate the collection."""
        self.logger.warning(f"⚠ Resetting collection '{collection_name}'")
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("Collection reset complete — 0 documents")


# ──────────────────────────────────────────────
# BM25 Index Builder
# ──────────────────────────────────────────────


class BM25Index:
    """
    Manages the BM25 sparse retrieval index.
    Built alongside the vector store during ingestion for hybrid search.

    Tokenizes chunks using code_aware_tokenize() which handles both
    prose (stopword removal, whitespace splitting) and code (camelCase,
    underscore, dot splitting).
    """

    def __init__(self, index_path: Path = config.BM25_INDEX_PATH):
        self.logger = logging.getLogger("pdf_chatbot_v2.bm25")
        self.index_path = index_path
        self.corpus_tokens: list[list[str]] = []
        self.chunk_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[dict]):
        """
        Build BM25 index from ingested chunks.

        Args:
            chunks: List of chunk dicts with "text" and "id" keys.
        """
        self.logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        self.corpus_tokens = []
        self.chunk_ids = []

        for chunk in chunks:
            tokens = code_aware_tokenize(chunk["text"])
            self.corpus_tokens.append(tokens)
            self.chunk_ids.append(chunk["id"])

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self.logger.info(
                f"BM25 index built — {len(self.corpus_tokens)} documents, "
                f"avg {sum(len(t) for t in self.corpus_tokens) / len(self.corpus_tokens):.0f} tokens/doc"
            )
        else:
            self.logger.warning("No chunks to build BM25 index from.")

    def save(self):
        """Persist BM25 index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_tokens": self.corpus_tokens,
            "chunk_ids": self.chunk_ids,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"BM25 index saved to: {self.index_path}")

    @classmethod
    def load(cls, index_path: Path = config.BM25_INDEX_PATH) -> "BM25Index":
        """Load a previously saved BM25 index from disk."""
        instance = cls(index_path)
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        instance.corpus_tokens = data["corpus_tokens"]
        instance.chunk_ids = data["chunk_ids"]
        if instance.corpus_tokens:
            instance.bm25 = BM25Okapi(instance.corpus_tokens)
        return instance


# ──────────────────────────────────────────────
# Main Ingestion Pipeline
# ──────────────────────────────────────────────


def run_ingestion(
    pdf_dir: Path = config.PDF_DIR,
    batch_size: int = config.BATCH_SIZE,
    reset: bool = False,
    enable_ocr: bool = config.ENABLE_OCR,
) -> dict:
    """
    Execute the full ingestion pipeline:
      PDF discovery → batch processing → extract → chunk → embed → store (vector + BM25)

    Args:
        pdf_dir: Directory containing PDF files.
        batch_size: Number of PDFs to process per batch.
        reset: If True, wipe the vector store before ingestion.
        enable_ocr: If True, enable RapidOCR fallback for scanned pages.

    Returns:
        Stats dict with counts and timings.
    """
    logger = logging.getLogger("pdf_chatbot_v2.ingest")
    start_time = time.time()

    # ── Stats tracking ──
    stats = {
        "total_pdfs": 0,
        "processed_pdfs": 0,
        "skipped_pdfs": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "total_time": 0.0,
        "content_types": {"code": 0, "prose": 0, "mixed": 0, "table": 0},
        "errors": [],
    }

    # ── Step 1: Discover PDFs ──
    logger.info(f"📂 Scanning for PDFs in: {pdf_dir}")
    try:
        pdf_files = discover_pdfs(pdf_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return stats

    stats["total_pdfs"] = len(pdf_files)
    logger.info(f"📄 Found {len(pdf_files)} PDF files to process")

    # ── Step 2: Initialize components ──
    embedder = EmbeddingEngine()
    store = VectorStore()

    if reset:
        store.reset()
        # Also remove old BM25 index
        if config.BM25_INDEX_PATH.exists():
            config.BM25_INDEX_PATH.unlink()
            logger.info("BM25 index removed")

    # Collect all chunks across batches for BM25 index building
    all_chunks_for_bm25 = []

    # ── Step 3: Batch processing loop ──
    batch_num = 0
    total_batches = (len(pdf_files) + batch_size - 1) // batch_size

    for batch in batch_generator(pdf_files, batch_size):
        batch_num += 1
        batch_start = time.time()
        logger.info(
            f"\n{'─' * 50}\n"
            f"  Batch {batch_num}/{total_batches} — "
            f"{len(batch)} PDFs\n"
            f"{'─' * 50}"
        )

        batch_chunks = []

        # ── Extract & Chunk ──
        for pdf_path in tqdm(
            batch, desc=f"Batch {batch_num} — Extracting", unit="pdf"
        ):
            pages = extract_text_from_pdf(pdf_path, logger, enable_ocr=enable_ocr)
            if not pages:
                stats["skipped_pdfs"] += 1
                stats["errors"].append(f"No text: {pdf_path.name}")
                continue

            stats["processed_pdfs"] += 1
            stats["total_pages"] += len(pages)

            chunks = chunk_pages(pages)
            batch_chunks.extend(chunks)

            # Track content type distribution
            for chunk in chunks:
                ct = chunk["metadata"].get("content_type", "prose")
                stats["content_types"][ct] = stats["content_types"].get(ct, 0) + 1

        if not batch_chunks:
            logger.warning(
                f"Batch {batch_num} produced no chunks — skipping embed+store"
            )
            continue

        # ── Embed ──
        logger.info(
            f"🧬 Embedding {len(batch_chunks)} chunks on "
            f"[{embedder.device.upper()}]..."
        )
        embed_start = time.time()
        texts = [c["text"] for c in batch_chunks]
        embeddings = embedder.encode(texts)
        embed_time = time.time() - embed_start
        logger.info(
            f"   Embedding complete in {format_time(embed_time)} "
            f"({len(batch_chunks) / max(embed_time, 0.001):.0f} chunks/sec)"
        )

        # ── Store in ChromaDB ──
        ids = [c["id"] for c in batch_chunks]
        metadatas = [c["metadata"] for c in batch_chunks]
        upserted = store.upsert(ids, embeddings, texts, metadatas)
        stats["total_chunks"] += upserted

        # ── Collect for BM25 ──
        all_chunks_for_bm25.extend(batch_chunks)

        batch_time = time.time() - batch_start
        logger.info(
            f"   Batch {batch_num} complete: {upserted} chunks stored "
            f"in {format_time(batch_time)}"
        )

        # ── Memory cleanup between batches ──
        del batch_chunks, texts, embeddings, ids, metadatas
        gc.collect()

    # ── Step 4: Build & Save BM25 Index ──
    if all_chunks_for_bm25:
        bm25_index = BM25Index()
        bm25_index.build(all_chunks_for_bm25)
        bm25_index.save()
    else:
        logger.warning("No chunks available for BM25 index construction.")

    # ── Final Report ──
    stats["total_time"] = time.time() - start_time
    _print_report(stats, store.count(), logger)
    return stats


def _print_report(stats: dict, total_in_store: int, logger: logging.Logger):
    """Print a formatted ingestion summary."""
    logger.info("\n" + "=" * 60)
    logger.info("  📊 INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  PDFs Found       : {stats['total_pdfs']}")
    logger.info(f"  PDFs Processed   : {stats['processed_pdfs']}")
    logger.info(f"  PDFs Skipped     : {stats['skipped_pdfs']}")
    logger.info(f"  Pages Extracted  : {stats['total_pages']}")
    logger.info(f"  Chunks Created   : {stats['total_chunks']}")
    logger.info(f"  Total in Store   : {total_in_store}")
    logger.info(f"  Total Time       : {format_time(stats['total_time'])}")

    # Content type breakdown
    ct = stats.get("content_types", {})
    if any(ct.values()):
        logger.info(
            f"  Content Types    : "
            f"prose={ct.get('prose', 0)}, code={ct.get('code', 0)}, "
            f"mixed={ct.get('mixed', 0)}, table={ct.get('table', 0)}"
        )

    if stats["total_time"] > 0:
        rate = stats["processed_pdfs"] / stats["total_time"]
        logger.info(f"  Throughput       : {rate:.1f} PDFs/sec")

    if stats["errors"]:
        logger.info(f"\n  ⚠ Errors ({len(stats['errors'])}):") 
        for err in stats["errors"][:10]:  # Show first 10
            logger.info(f"    • {err}")
        if len(stats["errors"]) > 10:
            logger.info(f"    ... and {len(stats['errors']) - 10} more")

    logger.info("=" * 60)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF-Chatbot-V2 — Document Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ingest.py                      # Ingest from default directory\n"
            "  python ingest.py --pdf-dir ./my_pdfs   # Custom PDF directory\n"
            "  python ingest.py --reset               # Wipe store and re-ingest\n"
            "  python ingest.py --batch-size 5         # Smaller batches for low RAM\n"
        ),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=config.PDF_DIR,
        help=f"Directory containing PDF files (default: {config.PDF_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"PDFs per batch — lower for less RAM usage (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the vector store before ingesting",
    )

    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument(
        "--ocr",
        dest="ocr",
        action="store_true",
        default=None,
        help="Enable RapidOCR fallback for scanned PDFs",
    )
    ocr_group.add_argument(
        "--no-ocr",
        dest="ocr",
        action="store_false",
        default=None,
        help="Disable RapidOCR fallback for scanned PDFs",
    )

    args = parser.parse_args()

    # Determine enable_ocr value
    enable_ocr = args.ocr
    if enable_ocr is None:
        if sys.stdin.isatty():
            try:
                user_choice = (
                    input(
                        "Do you want to enable OCR fallback for scanned PDFs? "
                        "(y/n) [default: y]: "
                    )
                    .strip()
                    .lower()
                )
                enable_ocr = user_choice not in ["n", "no"]
            except (KeyboardInterrupt, EOFError):
                enable_ocr = config.ENABLE_OCR
        else:
            enable_ocr = config.ENABLE_OCR

    # Initialize logging
    logger = setup_logging()
    logger.info("PDF-Chatbot-V2 — Ingestion Pipeline Starting")
    logger.info(f"Config: chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}")
    logger.info(f"Config: embedding_model={config.EMBEDDING_MODEL}")
    logger.info(f"Config: batch_size={args.batch_size}")
    logger.info(f"Config: enable_ocr={enable_ocr}")
    logger.info(f"Config: hybrid_search={config.USE_HYBRID_SEARCH}")

    # Ensure PDF directory exists
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    result = run_ingestion(
        pdf_dir=args.pdf_dir,
        batch_size=args.batch_size,
        reset=args.reset,
        enable_ocr=enable_ocr,
    )

    # Exit with error code if nothing was processed
    if result["processed_pdfs"] == 0:
        logger.error("No PDFs were successfully processed. Exiting with error.")
        sys.exit(1)
