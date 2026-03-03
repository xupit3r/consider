(ns consider.llm-test
  (:require [clojure.test :refer :all]
            [consider.llm :refer :all]
            [clojure.spec.alpha :as s]
            [consider.specs.llm :as llm-spec]
            [clojure.string :as str]))

(deftest test-mock-llm
  (let [llm (make-mock-llm)]
    (testing "predict-candidates with defaults"
      (let [candidates (predict-candidates llm [])]
        (is (not (empty? candidates)))
        (is (validate-candidate-step (first candidates)))))
    
    (testing "score-step with defaults"
      (let [score (score-step llm [] "Some Action")]
        (is (contains? score :pragmatic-estimate))
        (is (contains? score :epistemic-estimate))))))

(deftest test-prompt-templates
  (let [ctx [{:role :user :content "Ask about the world"}]]
    (testing "prediction-prompt"
      (let [prompt (prediction-prompt ctx)]
        (is (str/includes? prompt "USER: Ask about the world"))
        (is (str/includes? prompt "JSON array"))))
    
    (testing "scoring-prompt"
      (let [prompt (scoring-prompt ctx "Action 1")]
        (is (str/includes? prompt "USER: Ask about the world"))
        (is (str/includes? prompt "Candidate Action: Action 1"))
        (is (str/includes? prompt "JSON object"))))))

(deftest test-mock-llm-with-responses
  (let [ctx [{:role :user :content "Hello"}]
        resp [{:candidate-action "Greeting"
               :prior-prob 0.9
               :pragmatic-estimate 0.8
               :epistemic-estimate 0.2
               :confidence 1.0}]
        llm (make-mock-llm {ctx resp})]
    (testing "predict-candidates with custom responses"
      (is (= resp (predict-candidates llm ctx))))))
