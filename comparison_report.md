# PDF-Chatbot-V2 — Evaluation & Verification Report

**Timestamp:** 2026-08-22 10:37:40
**Model:** `qwen3:8b` (Context: 32768 tokens)
**Embedding:** `nomic-ai/nomic-embed-text-v1.5` (768-dim)
**Retrieval Strategy:** Hybrid (ChromaDB Vector + BM25 Sparse + RRF + Cross-Encoder)

---

## Domain-by-Domain Metrics

| # | Domain | Question | Precision | Faithfulness | Latency |
|---|---|---|---|---|---|
| 1 | General | What are the primary objectives and architect... | 100.0% | 20.0% | 513ms |
| 2 | General | What prerequisites, installations, and depend... | 100.0% | 80.0% | 256ms |
| 3 | Instructional | What are the step-by-step configuration instr... | 100.0% | 60.0% | 178ms |
| 4 | Instructional | What are the troubleshooting steps when encou... | 100.0% | 100.0% | 233ms |
| 5 | General | What security guidelines, permissions, or acc... | 100.0% | 100.0% | 49ms |
| 6 | Python | How does Python manage memory and garbage col... | 100.0% | 40.0% | 55ms |
| 7 | Python | Explain how Python decorators work and how to... | 100.0% | 0.0% | 50ms |
| 8 | Python | What is the difference between asyncio corout... | 100.0% | 20.0% | 64ms |
| 9 | Python | How do you handle exceptions properly using t... | 100.0% | 20.0% | 53ms |
| 10 | Python | How do generator functions and the yield keyw... | 100.0% | 0.0% | 59ms |
| 11 | Java | What is the difference between an abstract cl... | 100.0% | 40.0% | 49ms |
| 12 | Java | How does the Java Memory Model organize Heap,... | 100.0% | 20.0% | 63ms |
| 13 | Java | Explain the Java Collections Framework hierar... | 100.0% | 60.0% | 60ms |
| 14 | Java | How does Spring Boot dependency injection and... | 100.0% | 0.0% | 69ms |
| 15 | Java | What are Java Streams and how do lambda expre... | 100.0% | 60.0% | 50ms |
| 16 | DevOps | How do you structure a multi-stage Dockerfile... | 100.0% | 40.0% | 77ms |
| 17 | DevOps | Explain Kubernetes Pod lifecycle, ReplicaSets... | 100.0% | 60.0% | 60ms |
| 18 | DevOps | How do CI/CD automation pipelines trigger aut... | 100.0% | 60.0% | 106ms |
| 19 | DevOps | How do Terraform state files and declarative ... | 100.0% | 20.0% | 53ms |
| 20 | DevOps | How do Prometheus and Grafana collect and vis... | 100.0% | 40.0% | 56ms |
| 21 | Shell | How do you parse command line arguments using... | 100.0% | 20.0% | 114ms |
| 22 | Shell | Explain standard streams (stdin, stdout, stde... | 100.0% | 0.0% | 76ms |
| 23 | Shell | How do you implement error handling with 'set... | 100.0% | 20.0% | 69ms |
| 24 | Shell | How do awk, sed, and grep filter and transfor... | 100.0% | 40.0% | 90ms |
| 25 | Shell | How do background jobs, nohup, and subshells ... | 100.0% | 40.0% | 64ms |
| 26 | Fullstack | How do RESTful APIs handle versioning, status... | 100.0% | 100.0% | 60ms |
| 27 | Fullstack | Explain the React Component Virtual DOM lifec... | 100.0% | 20.0% | 50ms |
| 28 | Fullstack | How does CORS (Cross-Origin Resource Sharing)... | 100.0% | 0.0% | 56ms |
| 29 | Fullstack | Explain JWT (JSON Web Token) authentication f... | 100.0% | 20.0% | 53ms |
| 30 | Fullstack | How do WebSockets enable bidirectional, real-... | 100.0% | 0.0% | 49ms |

---

## Summary Aggregate Metrics

- **Average Context Precision:** `100.0%`
- **Average Faithfulness:** `36.7%`
- **Average Retrieval Latency:** `94.5 ms`
- **Total Benchmark Queries:** `30` across 6 technical domains

### Architectural Upgrades Verified

- [x] Hybrid BM25 + Dense vector fusion active (eliminates exact keyword misses)
- [x] Structure-aware chunking preserving code blocks, procedures, and tables
- [x] Lightweight ONNX-based RapidOCR fallback enabled
- [x] 32K context token window configured for Qwen3:8B