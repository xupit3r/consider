# Epistemic Foraging: Implementation Specs

## What This Is

These specs describe work needed to complete the **web foraging** extension to the Consider Active Inference agent. The goal: an autonomous agent that crawls the web, extracts knowledge into a graph, and uses Active Inference's curiosity drive (Expected Free Energy) to decide what to learn next.

## Current State

Scaffolding exists for all modules. Dependencies are added (`asami`, `jsoup`). Source files, tests, and Clojure specs exist. However:

- **extractor.clj has a runtime NPE** that blocks 3 tests
- **forager.clj has a stale return** (bare `obs-vector` expression before the actual return)
- **No integration test** ties the full loop together
- **Sleep consolidation** is stubbed but incomplete (no entity merging via LLM)
- **Search integration** (Phase 7) is not started

## Spec Index

| Spec | Module | Status | Priority |
|---|---|---|---|
| [01-fix-extractor](01-fix-extractor.md) | `web/extractor.clj` | Bug fix needed | P0 |
| [02-fix-forager-return](02-fix-forager-return.md) | `web/forager.clj` | Bug fix needed | P0 |
| [03-crawler-hardening](03-crawler-hardening.md) | `web/crawler.clj` | Enhancement | P1 |
| [04-graph-improvements](04-graph-improvements.md) | `web/graph.clj` | Enhancement | P1 |
| [05-knowledge-extraction](05-knowledge-extraction.md) | `web/knowledge.clj` | Enhancement | P1 |
| [06-ai-bridge](06-ai-bridge.md) | `world_model.clj`, `executive.clj`, `core.clj` | Stabilize | P1 |
| [07-sleep-consolidation](07-sleep-consolidation.md) | `web/forager.clj`, `web/graph.clj` | New work | P2 |
| [08-search-integration](08-search-integration.md) | `web/forager.clj`, `web/crawler.clj` | New work | P2 |
| [09-integration-test](09-integration-test.md) | `test/consider/web/` | New work | P1 |

## How To Run Tests

```bash
# All web tests
clojure -M:test -n consider.web.crawler-test -n consider.web.extractor-test -n consider.web.knowledge-test -n consider.web.graph-test -n consider.web.forager-test

# Full suite (includes existing core tests)
clojure -M:test
```

## Key Architecture Decisions Already Made

1. **Observation vector is 6-dim**: `[n-new-entities, n-confirmed, n-contradictions, n-new-relations, topic-similarity, page-quality]`
2. **Asami is the graph store** (in-memory, schema-less, Datalog queries)
3. **LLM extraction uses a `completion-fn`** (prompt-string -> response-string), not the protocol directly — this decouples from Ollama
4. **Crawler state is immutable** (functional — returns new state from every operation)
5. **Frontier is sorted by EFE score** (lowest = highest priority, consistent with minimizing G)
6. **The forager is separate from `core/step`** — it wraps the AI loop, not the other way around
