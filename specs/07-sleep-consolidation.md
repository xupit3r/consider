# Spec 07: Sleep & Consolidation

## Priority: P2

## Current State

`forager.clj/sleep-consolidate` is stubbed: it finds knowledge gaps and generates queries, but doesn't do entity merging or proper gap analysis.

`graph.clj/merge-entities!` exists but is untested and doesn't use LLM for canonicalization.

## Work Items

### 7a. LLM-driven entity merging

During sleep, the agent should:

1. Find entities that may be synonyms (e.g., "AI" and "Artificial Intelligence", "Karl Friston" and "K. Friston")
2. Ask the LLM to canonicalize them
3. Merge in the graph

```clojure
(defn consolidate-entities
  "Sleep phase: find and merge synonym entities.
   Uses LLM to determine which entities are the same."
  [forager-state]
  (let [{:keys [knowledge-graph llm-completion-fn]} forager-state
        entities (graph/query-all-entities knowledge-graph)
        entity-names (mapv first entities)
        ;; Group potential synonyms by similarity (simple: shared words)
        candidates (find-merge-candidates entity-names)
        ;; Ask LLM for each group
        merges (when llm-completion-fn
                 (for [group candidates]
                   (let [canonical (knowledge/canonicalize-entities
                                   llm-completion-fn group entity-names)]
                     canonical)))]
    ;; Apply merges
    (doseq [{:keys [old-name canonical-name]} merges]
      (graph/merge-entities! knowledge-graph old-name canonical-name))
    forager-state))
```

### 7b. Gap-driven query generation

Improve `sleep-consolidate` to:

1. Identify the top-N knowledge gaps (entities with fewest connections)
2. For each gap, find what questions would help (e.g., "Entity X has no relations — what is it related to?")
3. Generate search queries and seed the frontier

### 7c. Confidence decay

During sleep, triple confidence should decay slightly for unconfirmed triples:
- Triples seen from multiple sources: confidence stays high
- Triples from a single source: confidence decays by 5% per sleep cycle
- Below threshold (0.3): mark for re-verification

### 7d. Test `merge-entities!`

The existing function in `graph.clj` needs a test:

```clojure
(deftest test-merge-entities
  (let [kg (graph/make-knowledge-graph)]
    (graph/transact-entity! kg {:entity-name "AI" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "Artificial Intelligence" :entity-type "Concept"})
    (graph/transact-triple! kg {:subject "AI" :predicate "is-a" :object "Field"})
    (graph/transact-triple! kg {:subject "Researchers" :predicate "study" :object "AI"})
    (let [result (graph/merge-entities! kg "AI" "Artificial Intelligence")]
      (is (= 2 (:triples-rewritten result)))
      ;; Verify new triples reference canonical name
      (let [triples (graph/query-triples-about kg "Artificial Intelligence")]
        (is (>= (count triples) 2))))))
```

## Files

- `src/consider/web/forager.clj` (update `sleep-consolidate`)
- `src/consider/web/graph.clj` (confidence decay, merge testing)
- `src/consider/web/knowledge.clj` (add `canonicalize-entities`)
- `test/consider/web/graph_test.clj`
- `test/consider/web/forager_test.clj`

## Acceptance Criteria

- `sleep-consolidate` merges synonym entities when LLM is available
- `merge-entities!` correctly rewrites triples (tested)
- Gap-driven queries get added to the frontier
- Confidence decay runs during sleep
