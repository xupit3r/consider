# Spec 09: End-to-End Integration Test

## Priority: P1

## Current State

Each module has unit tests, but there's no test that runs the full foraging loop: seed -> fetch -> extract -> store -> score -> repeat.

## Work Items

### 9a. Mock HTTP layer

Tests must not make real network calls. Create a mock HTTP layer:

```clojure
(ns consider.web.test-helpers
  (:require [consider.web.crawler :as crawler]))

(def mock-pages
  "A map of URL -> {:status 200 :body \"<html>...\" :content-type \"text/html\"}"
  {"https://en.wikipedia.org/wiki/Active_Inference"
   {:status 200
    :body "<html><head><title>Active Inference</title></head>
           <body><article>
             <p>Active inference is a theory developed by Karl Friston at UCL.
                It is based on the free energy principle.</p>
             <p>Key concepts include variational inference and Markov blankets.</p>
             <a href='https://en.wikipedia.org/wiki/Free_energy_principle'>Free Energy Principle</a>
             <a href='https://en.wikipedia.org/wiki/Karl_Friston'>Karl Friston</a>
             <a href='https://en.wikipedia.org/wiki/Variational_Bayesian_methods'>Variational methods</a>
           </article></body></html>"
    :content-type "text/html"}
   ;; ... more pages
   })
```

Option A: Override `fetch-page` with a version that looks up from the map.
Option B: Use `with-redefs` in tests to mock `hato.client/get`.

Option B is cleaner:

```clojure
(defn with-mock-http [pages test-fn]
  (with-redefs [hato.client/get (fn [url opts]
                                   (let [page (get pages url)]
                                     (if page
                                       {:status (:status page)
                                        :body (:body page)
                                        :headers {"content-type" (:content-type page)}}
                                       {:status 404 :body "Not Found"})))]
    (test-fn)))
```

### 9b. Full loop test (no LLM)

Test the foraging loop with mock HTTP and no LLM (fallback entity extraction from links):

```clojure
(deftest test-foraging-loop-no-llm
  (with-mock-http mock-pages
    (fn []
      (let [forager (-> (forager/make-forager {})
                        (forager/seed-from-wikipedia "Active Inference"))
            result (forager/run-foraging forager 3)]
        ;; Verify pages were fetched
        (is (>= (get-in result [:stats :pages-fetched]) 1))
        ;; Verify entities extracted from links
        (is (pos? (get-in result [:stats :entities-extracted])))
        ;; Verify frontier grew
        (is (pos? (crawler/frontier-size (:crawler-state result))))
        ;; Verify graph has data
        (let [stats (graph/graph-stats (:knowledge-graph result))]
          (is (pos? (:entity-count stats))))))))
```

### 9c. Full loop test (with mock LLM)

Same as above but with a mock completion function that returns realistic entity/relation extraction:

```clojure
(deftest test-foraging-loop-with-llm
  (with-mock-http mock-pages
    (fn []
      (let [forager (-> (forager/make-forager
                         {:llm-completion-fn mock-completion-fn
                          :knowledge-goals ["active inference"]})
                        (forager/seed-from-wikipedia "Active Inference"))
            result (forager/run-foraging forager 5)]
        ;; With LLM, we should get real entities and triples
        (let [stats (graph/graph-stats (:knowledge-graph result))]
          (is (>= (:entity-count stats) 3))
          (is (>= (:triple-count stats) 1)))
        ;; Observation vector should be 6-dim
        (let [obs (forager/forager->observation result)]
          (is (= 6 (count obs)))
          (is (every? number? obs)))))))
```

### 9d. Sleep cycle test

```clojure
(deftest test-sleep-after-foraging
  (with-mock-http mock-pages
    (fn []
      (let [forager (-> (forager/make-forager {:knowledge-goals ["active inference"]})
                        (forager/seed-from-wikipedia "Active Inference"))
            after-foraging (forager/run-foraging forager 3)
            after-sleep (forager/sleep-consolidate after-foraging)]
        ;; Sleep should identify gaps
        (is (some? (:last-sleep-gaps after-sleep)))))))
```

### 9e. AI bridge integration test

Test the full pipeline: forager produces observation -> `core/step` runs -> action parsed -> forager executes action:

```clojure
(deftest test-forager-ai-loop
  (with-mock-http mock-pages
    (fn []
      (let [mock-llm (llm/make-mock-llm)
            agent (core/initialize-foraging-agent
                   {:knowledge-goals ["active inference"]
                    :llm-system mock-llm})
            forager (-> (forager/make-forager {})
                        (forager/seed-from-wikipedia "Active Inference"))
            ;; Step 1: forage
            [forager-1 obs-1] (forager/forage-step forager)
            ;; Step 2: run AI step with observation
            ai-result (when obs-1
                        (core/step agent obs-1
                                   {:inference-steps 5
                                    :reasoning-iterations 2
                                    :exploration-weight 1.0}))]
        (when obs-1
          (is (some? (:next-action ai-result)))
          (is (some? (:belief-state ai-result))))))))
```

## Files

- New: `test/consider/web/test_helpers.clj`
- New: `test/consider/web/integration_test.clj`

## Acceptance Criteria

- All integration tests pass with mock HTTP (no network calls)
- Full loop: seed -> fetch -> extract -> store -> observe -> AI step -> repeat
- Coverage of both LLM and no-LLM modes
- Sleep consolidation tested
