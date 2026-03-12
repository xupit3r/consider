(ns consider.web.ai-bridge-test
  (:require [clojure.test :refer :all]
            [consider.core :as core]
            [consider.world-model :as wm]
            [consider.executive :as exec]
            [consider.llm :as llm]
            [consider.web.forager :as forager]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]))

(deftest test-foraging-agent-step
  (testing "6a: Test initialize-foraging-agent + core/step"
    (let [mock-llm (llm/make-mock-llm)
          agent (core/initialize-foraging-agent
                 {:knowledge-goals ["active inference"]
                  :llm-system mock-llm})
          ;; Simulate a web page observation: 6-dim knowledge vector
          ;; [n-new-entities, n-confirmed, n-contradictions, n-new-relations, topic-similarity, page-quality]
          obs [3.0 1.0 0.0 5.0 0.8 0.7]
          result (core/step agent obs {:inference-steps 5
                                       :reasoning-iterations 3
                                       :exploration-weight 1.0})]
      (is (some? (:belief-state result)))
      (is (some? (:next-action result))))))

(deftest test-web-action-predictor
  (testing "6d: Test WebActionPredictor"
    (let [frontier-fn (fn [] [{:url "https://example.com/a" :efe-score 0.3}
                              {:url "https://example.com/b" :efe-score 0.7}])
          gap-fn (fn [] ["quantum mechanics" "neural networks"])
          predictor (exec/make-web-action-predictor frontier-fn gap-fn)
          candidates (llm/predict-candidates predictor [{:role :user :content "test"}])]
      (is (>= (count candidates) 3)) ;; 2 URLs + 2 gaps, but take 3 + take 2
      (is (every? :candidate-action candidates))
      (is (some #(clojure.string/starts-with? (:candidate-action %) "VISIT:") candidates))
      (is (some #(clojure.string/starts-with? (:candidate-action %) "SEARCH:") candidates)))))
