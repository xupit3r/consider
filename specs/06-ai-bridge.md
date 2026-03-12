# Spec 06: Active Inference Bridge Stabilization

## Priority: P1

## Current State

Three existing modules were modified to support web foraging:

- **`world_model.clj`**: Added `make-knowledge-slot`, `knowledge-likelihood-fn`, `knowledge-transition-fn`
- **`executive.clj`**: Added `WebActionPredictor` record implementing `PolicyPredictor`
- **`core.clj`**: Added `initialize-foraging-agent`

These compile but haven't been tested as an integrated system. The key question is whether `core/step` works when the observation is a 6-dim knowledge vector and actions are URLs.

## Work Items

### 6a. Test `initialize-foraging-agent` + `core/step`

Write a test that:

```clojure
(deftest test-foraging-agent-step
  (let [mock-llm (llm/make-mock-llm)
        agent (core/initialize-foraging-agent
               {:knowledge-goals ["active inference"]
                :llm-system mock-llm})
        ;; Simulate a web page observation
        obs [3.0 1.0 0.0 5.0 0.8 0.7]  ;; 3 new entities, 1 confirmed, 5 new relations, etc.
        result (core/step agent obs {:inference-steps 5
                                     :reasoning-iterations 3
                                     :exploration-weight 1.0})]
    (is (some? (:belief-state result)))
    (is (some? (:next-action result)))))
```

This will likely surface issues with:
- Dimension mismatches between slot positions (6-dim) and observation (6-dim)
- The likelihood function returning the wrong shape
- The transition function not handling string actions

### 6b. Fix `knowledge-likelihood-fn`

Currently it computes predicted observations from slot positions. Verify it returns exactly 6 doubles that are compatible with the VFE calculation in `inference.clj`. The VFE calculation creates Neanderthal vectors from the result, so each element must be a valid double.

### 6c. Fix `knowledge-transition-fn`

Currently it only reduces variance. It should also update position based on the observation:

```clojure
(defn knowledge-transition-fn []
  (fn [internal-states action]
    (reduce-kv (fn [acc id slot]
                 (let [pos (:position slot)
                       var (:variance slot)
                       ;; Variance decays with each observation
                       new-var (mapv #(max 0.01 (* % 0.95)) var)
                       ;; Position shifts slightly toward center (mean-reversion)
                       ;; Actual update happens in belief-update via flow matching
                       ]
                   (assoc acc id (assoc slot :variance new-var))))
               {}
               internal-states)))
```

### 6d. Test `WebActionPredictor`

```clojure
(deftest test-web-action-predictor
  (let [frontier-fn (fn [] [{:url "https://example.com/a" :efe-score 0.3}
                             {:url "https://example.com/b" :efe-score 0.7}])
        gap-fn (fn [] ["quantum mechanics" "neural networks"])
        predictor (exec/make-web-action-predictor frontier-fn gap-fn)
        candidates (llm/predict-candidates predictor [{:role :user :content "test"}])]
    (is (>= (count candidates) 3))  ;; 2 URLs + 2 gaps, but take 3 + take 2
    (is (every? :candidate-action candidates))
    (is (some #(str/starts-with? (:candidate-action %) "VISIT:") candidates))
    (is (some #(str/starts-with? (:candidate-action %) "SEARCH:") candidates))))
```

### 6e. Action parsing in forager

When `core/step` returns a `next-action`, the forager needs to parse it:
- `"VISIT:https://..."` -> fetch that URL
- `"SEARCH:query terms"` -> formulate search URL and fetch
- Anything else -> ignore

This parsing doesn't exist yet. Add to `forager.clj`:

```clojure
(defn parse-action [action-str]
  (cond
    (str/starts-with? action-str "VISIT:") {:type :visit :url (subs action-str 6)}
    (str/starts-with? action-str "SEARCH:") {:type :search :query (subs action-str 7)}
    :else {:type :unknown :raw action-str}))
```

## Files

- `src/consider/world_model.clj`
- `src/consider/executive.clj`
- `src/consider/core.clj`
- `src/consider/web/forager.clj`
- New: `test/consider/web/ai_bridge_test.clj`

## Acceptance Criteria

- `core/step` completes without error when given a 6-dim knowledge observation
- `WebActionPredictor` generates valid candidates
- Action parsing round-trips correctly
- No regressions in existing `core_test.clj`
