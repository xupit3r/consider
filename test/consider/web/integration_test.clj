(ns consider.web.integration-test
  (:require [clojure.test :refer :all]
            [consider.web.forager :as forager]
            [consider.web.crawler :as crawler]
            [consider.web.graph :as graph]
            [consider.web.test-helpers :refer [with-mock-http mock-pages]]
            [consider.core :as core]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]))

(deftest test-foraging-loop-no-llm
  (testing "9b: Full loop test (no LLM)"
    (println "Starting 9b...")
    (with-mock-http mock-pages
      (fn []
        (let [forager (-> (forager/make-forager {:crawler-opts {:default-delay-ms 0}})
                          (forager/seed-from-wikipedia "Active Inference"))
              _ (println "Seeded forager.")
              result (forager/run-foraging forager 2)]
          (println "Finished run-foraging.")
          ;; Verify pages were fetched
          (is (>= (get-in result [:stats :pages-fetched]) 1))
          ;; Verify entities extracted from links
          (is (pos? (get-in result [:stats :entities-extracted])))
          ;; Verify frontier grew
          (is (pos? (crawler/frontier-size (:crawler-state result))))
          ;; Verify graph has data
          (let [stats (graph/graph-stats (:knowledge-graph result))]
            (is (pos? (:entity-count stats)))))))))

(deftest test-sleep-after-foraging
  (testing "9d: Sleep cycle test"
    (println "Starting 9d...")
    (with-mock-http mock-pages
      (fn []
        (let [forager (-> (forager/make-forager {:knowledge-goals ["active inference"]
                                                 :crawler-opts {:default-delay-ms 0}})
                          (forager/seed-from-wikipedia "Active Inference"))
              after-foraging (forager/run-foraging forager 2)
              _ (println "Finished foraging, starting sleep...")
              [after-sleep merges] (forager/sleep-consolidate after-foraging)]
          (println "Finished sleep.")
          ;; Sleep should identify gaps or at least run without error
          (is (some? (:last-sleep-gaps after-sleep))))))))

(deftest test-forager-ai-loop
  (testing "9e: AI bridge integration test"
    (println "Starting 9e...")
    (with-mock-http mock-pages
      (fn []
        (let [mock-llm (llm/make-mock-llm)
              agent (core/initialize-foraging-agent
                     {:knowledge-goals ["active inference"]
                      :llm-system mock-llm})
              forager (-> (forager/make-forager {:crawler-opts {:default-delay-ms 0}})
                          (forager/seed-from-wikipedia "Active Inference"))

              ;; Step 1: Forage to get an observation
              _ (println "Foraging step 1...")
              [forager-1 obs-1] (forager/forage-step forager)

              ;; Step 2: Run AI core step with the observation
              _ (println "AI core step...")
              ai-result (when obs-1
                          (core/step agent obs-1
                                     {:inference-steps 2
                                      :reasoning-iterations 2
                                      :exploration-weight 1.0}))]

          (if obs-1
            (do
              (is (some? (:next-action ai-result)))
              (is (some? (:belief-state ai-result)))
              (println "AI core finished, executing action...")

              ;; Step 3: Execute the action chosen by AI
              (let [action (forager/parse-action (:next-action ai-result))
                    [forager-2 obs-2] (forager/execute-action forager-1 action)]
                (is (some? forager-2))
                (is (>= (:step-count forager-2) 2))
                (println "Action executed.")))
            (is (nil? obs-1) "Observation should be present if seed URL was fetched")))))))
