"""
PDF-Chatbot-V2 — Master Pipeline Launcher (Train & Chat)
========================================================
All-in-one entry point that:
  1. Verifies compute hardware (CUDA GPU / CPU) & Ollama connection
  2. Runs full document ingestion & index generation (ChromaDB + BM25)
  3. Seamlessly launches the interactive query terminal

Usage:
    python main.py                     # Ingest if needed / updated, then launch chat
    python main.py --reset             # Force full re-ingestion from scratch, then chat
    python main.py --skip-ingest       # Skip ingestion and open chat directly
    python main.py --no-ocr            # Ingest text-only without OCR fallback
    python main.py --batch-size 20     # Process 20 PDFs per batch
    python main.py --top-k 8           # Set top-K retrieved context chunks
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

import config
from ingest import run_ingestion
from query import ChatEngine
from utils import detect_device, format_time, setup_logging

console = Console()


def display_welcome_banner():
    """Display startup header with pipeline details."""
    console.print(
        Panel(
            "[bold cyan]PDF-Chatbot-V2 — End-to-End Pipeline Launcher[/bold cyan]\n"
            "Lightweight, High-Performance Hybrid RAG for General & Technical Documents\n"
            f"[dim]Generation: {config.OLLAMA_MODEL} ({config.OLLAMA_CONTEXT_WINDOW} ctx) | "
            f"Embeddings: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIMENSION}d) | "
            f"Retrieval: Hybrid (BM25 + Vector + RRF)[/dim]",
            border_style="cyan",
        )
    )


def should_run_ingestion(reset_flag: bool, skip_flag: bool) -> bool:
    """Determine whether to run ingestion based on flags and vectorstore existence."""
    if skip_flag:
        return False
    if reset_flag:
        return True

    # If vector store or BM25 index doesn't exist or is empty, run ingestion
    chroma_dir = config.VECTORSTORE_DIR
    bm25_file = config.BM25_INDEX_PATH

    if not chroma_dir.exists() or not bm25_file.exists():
        return True

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(name=config.COLLECTION_NAME)
        if collection.count() == 0:
            return True
    except Exception:
        return True

    # If already indexed, ask user or proceed
    console.print(
        f"[green]✔ Existing index detected at '{chroma_dir}' "
        f"({collection.count()} chunks indexed).[/green]"
    )
    if sys.stdin.isatty():
        try:
            choice = input("Do you want to re-run ingestion/training on PDFs? (y/N): ").strip().lower()
            return choice in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False
    return False


def main():
    parser = argparse.ArgumentParser(
        description="PDF-Chatbot-V2 — Full Pipeline Launcher (Train & Chat)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                     # Normal startup (check index -> chat)\n"
            "  python main.py --reset             # Wipe index, re-train on all PDFs, then chat\n"
            "  python main.py --skip-ingest       # Skip training and jump straight to chat\n"
            "  python main.py --no-ocr            # Train faster without OCR fallback\n"
            "  python main.py --batch-size 20     # Custom batch size\n"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing vector store and BM25 index and re-ingest all PDFs",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip document ingestion/training step and launch chat directly",
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
        help=f"PDFs processed per batch (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of retrieved context chunks (default: {config.TOP_K})",
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

    config.ensure_directories()
    setup_logging()
    display_welcome_banner()

    # Step 1: Hardware detection
    device = detect_device()

    # Step 2: Document Ingestion / Training
    if should_run_ingestion(args.reset, args.skip_ingest):
        enable_ocr = args.ocr if args.ocr is not None else config.ENABLE_OCR
        console.print(
            Panel(
                f"[bold yellow]Step 1/2: Document Ingestion & Training[/bold yellow]\n"
                f"Scanning directory: [bold cyan]{args.pdf_dir}[/bold cyan] (recursive)\n"
                f"Batch Size: {args.batch_size} | OCR Enabled: {enable_ocr} | Device: [{device.upper()}]",
                border_style="yellow",
            )
        )
        t0 = time.time()
        stats = run_ingestion(
            pdf_dir=args.pdf_dir,
            batch_size=args.batch_size,
            reset=args.reset,
            enable_ocr=enable_ocr,
        )
        elapsed = time.time() - t0
        console.print(
            f"[bold green]✔ Training Complete![/bold green] "
            f"Indexed {stats['total_chunks']} chunks from {stats['processed_pdfs']} PDFs "
            f"in {format_time(elapsed)}.\n"
        )
    else:
        console.print("[dim]Skipping ingestion step — using existing index.[/dim]\n")

    # Step 3: Launch Interactive Terminal
    console.print(
        Panel(
            "[bold green]Step 2/2: Launching Interactive Query Terminal[/bold green]\n"
            "Type your questions naturally. Use [bold cyan]/help[/bold cyan] for available commands or [bold red]/quit[/bold red] to exit.",
            border_style="green",
        )
    )

    chat = ChatEngine(top_k=args.top_k)
    chat.run_interactive()


if __name__ == "__main__":
    main()
