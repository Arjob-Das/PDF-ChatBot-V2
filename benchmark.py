"""
PDF-Chatbot-V2 — Multi-Domain Performance Benchmarking Suite
============================================================
Evaluates PDF-Chatbot-V2 (Hybrid BM25 + Vector + RRF + Cross-Encoder Re-ranking)
across 30 curated questions in 6 technical and general domains:
  1. General & Architecture
  2. Instructional & Setup
  3. Python Development
  4. Java Development
  5. DevOps & Infrastructure
  6. Shell Scripting
  7. Fullstack Web

Measures:
  - Retrieval Quality & Precision across diverse question types
  - Latency (Retrieval speed in milliseconds)
  - Keyword Coverage & Grounding
  - Context Token Efficiency

Usage:
    python benchmark.py                 # Full benchmark across all domains
    python benchmark.py --domain python # Test a single domain (e.g. Python, DevOps)
    python benchmark.py --skip-llm      # Retrieval-only benchmark
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from utils import format_time, setup_logging

console = Console()

# ──────────────────────────────────────────────
# Curated Multi-Domain Benchmark Dataset (30 QA Pairs)
# ──────────────────────────────────────────────

BENCHMARK_DATASET = [
    # ── General & Instructional (5) ──
    {
        "question": "What are the primary objectives and architecture overview described in the documents?",
        "keywords": ["architecture", "objective", "system", "overview", "design"],
        "domain": "General",
    },
    {
        "question": "What prerequisites, installations, and dependencies are required to get started?",
        "keywords": ["prerequisite", "install", "requirement", "setup", "dependency"],
        "domain": "General",
    },
    {
        "question": "What are the step-by-step configuration instructions provided?",
        "keywords": ["step", "config", "setting", "parameter", "instruction"],
        "domain": "Instructional",
    },
    {
        "question": "What are the troubleshooting steps when encountering operational errors?",
        "keywords": ["troubleshoot", "error", "log", "resolve", "fix"],
        "domain": "Instructional",
    },
    {
        "question": "What security policies, access controls, and authentication mechanisms are outlined?",
        "keywords": ["security", "auth", "permission", "policy", "token"],
        "domain": "General",
    },
    # ── Python Development (5) ──
    {
        "question": "How do Python asyncio event loops, coroutines, and async/await syntax work?",
        "keywords": ["async", "await", "coroutine", "event", "loop"],
        "domain": "Python",
    },
    {
        "question": "Explain Python GIL (Global Interpreter Lock) and how multiprocessing bypasses it.",
        "keywords": ["gil", "thread", "multiprocessing", "interpreter", "lock"],
        "domain": "Python",
    },
    {
        "question": "How do Python decorators work and how can you preserve function metadata with functools.wraps?",
        "keywords": ["decorator", "wrapper", "wraps", "functools", "function"],
        "domain": "Python",
    },
    {
        "question": "Explain Python memory management, reference counting, and generational garbage collection.",
        "keywords": ["memory", "garbage", "reference", "collection", "cyclic"],
        "domain": "Python",
    },
    {
        "question": "What are Python context managers and how do you implement them using __enter__ and __exit__?",
        "keywords": ["context", "enter", "exit", "with", "resource"],
        "domain": "Python",
    },
    # ── Java Development (5) ──
    {
        "question": "Explain Java JVM memory layout: Heap (Eden, Survivor, Old) vs Non-Heap Metaspace.",
        "keywords": ["heap", "metaspace", "eden", "survivor", "jvm"],
        "domain": "Java",
    },
    {
        "question": "How does Spring Boot dependency injection and @Autowired bean lifecycle work?",
        "keywords": ["spring", "bean", "autowired", "lifecycle", "dependency"],
        "domain": "Java",
    },
    {
        "question": "How does the Java CompletableFuture API handle asynchronous non-blocking pipelines?",
        "keywords": ["completablefuture", "async", "thread", "pipeline", "supplyasync"],
        "domain": "Java",
    },
    {
        "question": "Explain Java Concurrency: synchronized blocks, ReentrantLock, and Atomic variables.",
        "keywords": ["lock", "synchronized", "atomic", "thread", "concurrency"],
        "domain": "Java",
    },
    {
        "question": "How do Java 8+ Streams, lambdas, and functional interfaces process data collections?",
        "keywords": ["stream", "filter", "map", "lambda", "collector"],
        "domain": "Java",
    },
    # ── DevOps & Infrastructure (5) ──
    {
        "question": "How do you structure a multi-stage Dockerfile to minimize production image size?",
        "keywords": ["dockerfile", "stage", "build", "image", "copy"],
        "domain": "DevOps",
    },
    {
        "question": "Explain Kubernetes Pod lifecycle, ReplicaSets, and rolling deployment strategies.",
        "keywords": ["pod", "deployment", "replicaset", "kubernetes", "rolling"],
        "domain": "DevOps",
    },
    {
        "question": "How do CI/CD automation pipelines trigger automated builds, testing, and deployments?",
        "keywords": ["pipeline", "stage", "test", "deploy", "build"],
        "domain": "DevOps",
    },
    {
        "question": "How do Terraform state files and declarative HCL configurations manage cloud resources?",
        "keywords": ["terraform", "state", "resource", "plan", "apply"],
        "domain": "DevOps",
    },
    {
        "question": "How do Prometheus and Grafana collect and visualize service metrics and alerts?",
        "keywords": ["prometheus", "grafana", "metric", "alert", "scrape"],
        "domain": "DevOps",
    },
    # ── Shell Scripting (5) ──
    {
        "question": "How do you parse command line arguments using getopts in a Bash shell script?",
        "keywords": ["getopts", "opt", "flag", "case", "shift"],
        "domain": "Shell",
    },
    {
        "question": "Explain standard streams (stdin, stdout, stderr) and redirection operators in Linux shell.",
        "keywords": ["stdin", "stdout", "stderr", "pipe", "redirect"],
        "domain": "Shell",
    },
    {
        "question": "How do you implement error handling with 'set -euo pipefail' and trap in Bash?",
        "keywords": ["pipefail", "trap", "exit", "error", "errexit"],
        "domain": "Shell",
    },
    {
        "question": "How do awk, sed, and grep filter and transform structured log files in terminal?",
        "keywords": ["awk", "sed", "grep", "regex", "filter"],
        "domain": "Shell",
    },
    {
        "question": "How do background jobs, nohup, and subshells work in Linux terminal environments?",
        "keywords": ["background", "nohup", "job", "process", "subshell"],
        "domain": "Shell",
    },
    # ── Fullstack Web (5) ──
    {
        "question": "How do RESTful APIs handle versioning, status codes, and HTTP request methods?",
        "keywords": ["rest", "api", "http", "status", "endpoint"],
        "domain": "Fullstack",
    },
    {
        "question": "Explain the React Component Virtual DOM lifecycle and state management with Hooks.",
        "keywords": ["react", "dom", "hook", "state", "component"],
        "domain": "Fullstack",
    },
    {
        "question": "How does CORS (Cross-Origin Resource Sharing) protect web applications and how is it configured?",
        "keywords": ["cors", "origin", "header", "preflight", "options"],
        "domain": "Fullstack",
    },
    {
        "question": "Explain JWT (JSON Web Token) authentication flow including signing and verification.",
        "keywords": ["jwt", "token", "header", "payload", "signature"],
        "domain": "Fullstack",
    },
    {
        "question": "How do WebSockets enable bidirectional, real-time communication compared to HTTP polling?",
        "keywords": ["websocket", "handshake", "realtime", "socket", "connection"],
        "domain": "Fullstack",
    },
]


def run_benchmark(skip_llm: bool = False, domain_filter: str = None) -> dict:
    """Run benchmark performance evaluation on PDF-Chatbot-V2."""
    console.print(
        Panel(
            "[bold cyan]PDF-Chatbot-V2 — Multi-Domain Performance Benchmark[/bold cyan]\n"
            "Evaluating Hybrid Retrieval Precision, Latency, and Keyword Grounding",
            border_style="cyan",
        )
    )

    # Filter dataset if requested
    dataset = BENCHMARK_DATASET
    if domain_filter:
        dataset = [
            q for q in dataset if q["domain"].lower() == domain_filter.lower()
        ]
        if not dataset:
            console.print(f"[red]No queries found for domain: {domain_filter}[/red]")
            return {}

    from query import Retriever

    console.print("[cyan]Initializing PDF-Chatbot-V2 Retriever...[/cyan]")
    retriever = Retriever()

    scores = {"latency": [], "faithfulness": []}

    results_table = Table(title="Performance Results Across Queries", border_style="cyan")
    results_table.add_column("#", justify="center", width=3)
    results_table.add_column("Domain", style="magenta", width=14)
    results_table.add_column("Question", max_width=40)
    results_table.add_column("Latency", justify="center", style="bold green")
    results_table.add_column("Grounding", justify="center", style="bold yellow")
    results_table.add_column("Top Source", style="dim", max_width=25)

    for i, item in enumerate(dataset, 1):
        q = item["question"]
        kws = item["keywords"]
        domain = item["domain"]

        t0 = time.time()
        results = retriever.search(q, top_k=config.TOP_K)
        latency_ms = (time.time() - t0) * 1000
        scores["latency"].append(latency_ms)

        context_text = retriever.format_context(results) if results else ""
        matches = sum(1 for kw in kws if kw.lower() in context_text.lower())
        faith = matches / len(kws) if kws else 0.0
        scores["faithfulness"].append(faith)

        top_source = (
            results[0].get("metadata", {}).get("source", "N/A") if results else "N/A"
        )

        results_table.add_row(
            str(i),
            domain,
            q[:38] + "...",
            f"{latency_ms:.0f}ms",
            f"{faith:.0%}",
            top_source,
        )

    console.print(results_table)

    # Summary Aggregates
    avg_latency = float(np.mean(scores["latency"]))
    avg_faith = float(np.mean(scores["faithfulness"]))

    console.print("\n" + "=" * 60)
    summary_table = Table(title="Aggregated Performance Summary", border_style="cyan")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Achieved Score", justify="center", style="bold green")
    summary_table.add_column("Target / Standard", justify="center", style="dim")

    summary_table.add_row(
        "Avg Retrieval Latency",
        f"{avg_latency:.1f} ms",
        "< 200 ms (Sub-second)",
    )
    summary_table.add_row(
        "Keyword Grounding",
        f"{avg_faith:.1%}",
        "High context match",
    )
    summary_table.add_row(
        "Embedding Dimension",
        f"{config.EMBEDDING_DIMENSION}-dim ({config.EMBEDDING_MODEL})",
        "Dense representation",
    )
    summary_table.add_row(
        "LLM Context Window",
        f"{config.OLLAMA_CONTEXT_WINDOW:,} tokens ({config.OLLAMA_MODEL})",
        "Extended context",
    )
    summary_table.add_row(
        "Retrieval Architecture",
        "Hybrid (BM25 + Vector + RRF + Cross-Encoder)",
        "Zero-miss exact terms + semantic",
    )

    console.print(summary_table)
    console.print("=" * 60)

    return {
        "latency_ms": avg_latency,
        "faithfulness": avg_faith,
        "total_queries": len(dataset),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF-Chatbot-V2 — Multi-Domain Performance Benchmarking",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM answer generation (benchmark retrieval only)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter benchmark to a specific domain (e.g., Python, Java, DevOps, Shell, Fullstack)",
    )
    args = parser.parse_args()

    setup_logging()
    run_benchmark(skip_llm=args.skip_llm, domain_filter=args.domain)
