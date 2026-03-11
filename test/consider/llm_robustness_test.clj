(ns consider.llm-robustness-test
  (:require [clojure.test :refer :all]
            [consider.llm :as llm]
            [clojure.spec.alpha :as s]
            [consider.specs.llm :as llm-spec]))

(deftest test-robust-json-parsing
  (testing "Handling Markdown-wrapped JSON"
    (let [completion-fn (fn [_] "```json\n[{\"candidate-action\": \"TEST\", \"prior-prob\": 0.9, \"pragmatic-estimate\": 0.8, \"epistemic-estimate\": 0.1}]\n```")
          llm (llm/->DynamicLLM "test-model" "key" :provider completion-fn)]
      (let [candidates (llm/predict-candidates llm [])]
        (is (= 1 (count candidates)))
        (is (= "TEST" (:candidate-action (first candidates)))))))

  (testing "Handling missing keys with defaults"
    (let [completion-fn (fn [_] "[{\"candidate-action\": \"PARTIAL\"}]")
          llm (llm/->DynamicLLM "test-model" "key" :provider completion-fn)]
      (let [candidates (llm/predict-candidates llm [])]
        (is (= "PARTIAL" (:candidate-action (first candidates))))
        ;; Should provide defaults for missing math components
        (is (some? (:prior-prob (first candidates)))))))

  (testing "Handling out-of-bounds probabilities"
    (let [completion-fn (fn [_] "[{\"candidate-action\": \"OOB\", \"prior-prob\": 1.5, \"pragmatic-estimate\": -0.5, \"epistemic-estimate\": 0.1}]")
          llm (llm/->DynamicLLM "test-model" "key" :provider completion-fn)]
      (let [candidates (llm/predict-candidates llm [])
            cand (first candidates)]
        ;; Values should be clamped to [0, 1]
        (is (<= (:prior-prob cand) 1.0))
        (is (>= (:pragmatic-estimate cand) 0.0)))))

  (testing "Handling completely malformed garbage"
    (let [completion-fn (fn [_] "I am sorry, I cannot perform this action.")
          llm (llm/->DynamicLLM "test-model" "key" :provider completion-fn)]
      (let [candidates (llm/predict-candidates llm [])]
        (is (vector? candidates))
        (is (contains? (first candidates) :candidate-action))
        (is (= 0.0 (:prior-prob (first candidates))))))))
