# Spec 04: Knowledge Graph Improvements

## Priority: P1

## Current State

`src/consider/web/graph.clj` wraps Asami and passes all tests (20 entities, 100 triples, gap detection, embeddings). But it needs improvements for production foraging.

## Work Items

### 4a. Entity deduplication / upsert

Currently `transact-entity!` always inserts. If the same entity name is transacted twice, Asami may create duplicates or silently overwrite depending on `:id`. The `:id` is derived from entity name, so Asami should upsert. **Verify this behavior** with a test:

```clojure
(deftest test-entity-upsert
  (let [kg (graph/make-knowledge-graph)]
    (graph/transact-entity! kg {:entity-name "Foo" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "Foo" :entity-type "Theory"})
    ;; Should have 1 unique entity, type updated to "Theory"
    (is (= 1 (count (graph/query-entity kg "Foo"))))
    (is (= "Theory" (second (first (graph/query-entity kg "Foo")))))))
```

If Asami doesn't upsert correctly, add explicit check-before-insert logic.

### 4b. Confirmed-count increment

`transact-entity!` sets `:entity/confirmed-count 1` every time. On re-encounter, it should increment the count. This metric is used in the observation vector (confirmed vs new entities).

### 4c. Contradiction detection

The observation vector has a `n-contradictions` dimension (index 2) that's always 0. Implement basic contradiction detection:

- When transacting a triple, check if a contradicting triple exists (same subject+predicate, different object for functional predicates like "born-in", "capital-of")
- Return contradiction count from `transact-extraction!`
- Store contradictions as metadata on the triple

### 4d. Topic clustering for slot projection

The plan describes projecting the graph into Active Inference slots via topic clustering:

```
Asami graph → Topic clustering (causal.clj cluster-causal-modules) → Slots
```

Add a function:

```clojure
(defn project-to-slots
  "Projects the knowledge graph into Active Inference slots.
   Groups entities into topic clusters and computes position/variance for each.
   Returns a map of slot-id -> {:entity-id :position :variance}"
  [kg n-clusters]
  ...)
```

This should:
1. Build a co-occurrence matrix from triples (entities that appear in triples together are "connected")
2. Use the existing `causal/cluster-causal-modules` to group entities
3. Compute position = `knowledge-embedding` for the subgraph of each cluster
4. Compute variance = inversely proportional to connection density within cluster

### 4e. Graph persistence

Currently the graph is in-memory only (`asami:mem://`). Add optional file-backed persistence:

```clojure
(defn make-persistent-knowledge-graph
  "Creates a file-backed Asami knowledge graph."
  [path]
  (let [uri (str "asami:local://" path)]
    (d/create-database uri)
    {:uri uri
     :connection (d/connect uri)
     :entity-count (atom 0)
     :triple-count (atom 0)}))
```

## Files

- `src/consider/web/graph.clj`
- `test/consider/web/graph_test.clj`

## Acceptance Criteria

- Entity upsert behavior verified and working
- Contradiction detection returns counts
- `project-to-slots` returns valid slot maps
- All existing graph tests still pass
- New tests for each improvement
