# Spec 05: Knowledge Extraction Improvements

## Priority: P1

## Current State

`src/consider/web/knowledge.clj` implements two-stage KGGen extraction (entities then relations) via a `completion-fn`. Tests pass with a mock LLM. The `KnowledgeExtractor` protocol is defined here AND in `llm.clj` (duplication issue).

## Work Items

### 5a. Remove duplicate protocol definition

`KnowledgeExtractor` is defined in two places:
- `consider.web.knowledge` (4 methods: `extract-entities`, `extract-relations`, `canonicalize-entity`, `formulate-query`)
- `consider.llm` (2 methods: `extract-knowledge`, `formulate-query`)

**Decision needed**: Keep ONE. The `llm.clj` version is simpler and follows the existing pattern (protocols in `llm.clj`, implementations in `llm/ollama.clj`). Recommendation:

1. Remove the protocol from `knowledge.clj` — it should be a pure-functions module
2. Keep the protocol in `llm.clj` as-is
3. The `ollama.clj` `extend-type` implementation already delegates to `knowledge.clj` functions, which is correct

### 5b. Entity canonicalization function

`canonicalization-prompt` exists but there's no function that uses it. Add:

```clojure
(defn canonicalize-entities
  "Given a list of new entities and existing canonical names,
   returns entities with canonical names resolved via LLM fuzzy matching."
  [completion-fn new-entities existing-names]
  ...)
```

This is needed for sleep-phase entity merging (spec 07).

### 5c. Extraction robustness

The current prompts request JSON output. Common LLM failure modes:
- Returns narrative text instead of JSON
- Returns partial JSON
- Returns entities not present in the text (hallucination)
- Returns empty arrays

Improve `parse-entities-response` and `parse-triples-response`:
- Filter out entities whose names don't appear (case-insensitive substring match) in the source text
- Filter out triples where subject or object wasn't in the extracted entity list
- Handle empty extraction gracefully (return `{:entities [] :triples []}` not a default placeholder)

### 5d. Chunk-aware extraction

Currently `text-to-triples` takes a single text string. When the forager calls it per-chunk in a loop, entities from earlier chunks aren't available as context for later chunks' relation extraction. The forager already accumulates entities across chunks (see `forager.clj` lines 140-150), but this could be cleaner:

```clojure
(defn extract-from-chunks
  "Extracts entities and relations from multiple text chunks,
   accumulating entity context across chunks."
  [completion-fn chunks]
  ...)
```

This would replace the reduce loop in `forager.clj`.

## Files

- `src/consider/web/knowledge.clj`
- `src/consider/llm.clj` (protocol lives here)
- `src/consider/llm/ollama.clj` (implementation)
- `test/consider/web/knowledge_test.clj`

## Acceptance Criteria

- Single protocol definition (in `llm.clj`)
- `canonicalize-entities` function works with mock LLM
- Hallucination filtering removes entities not in source text
- `extract-from-chunks` tested with 3+ chunks
- All knowledge tests pass
