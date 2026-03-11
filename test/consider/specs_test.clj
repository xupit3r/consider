(ns consider.specs-test
  (:require [clojure.test :refer :all]
            [clojure.spec.alpha :as s]
            [consider.specs.world-model :as wm-spec]
            [consider.specs.llm :as llm-spec]))

(deftest test-world-model-specs
  (testing "Valid belief-state"
    (let [bs {:internal-states {:me {:entity-id :me :position [0.0] :variance [1.0]}}
              :variational-free-energy 0.5
              :expected-free-energy 1.0
              :efe-components {:risk 0.5 :ambiguity 0.5}
              :preferences []}]
      (is (s/valid? ::wm-spec/belief-state bs))))

  (testing "Invalid belief-state (missing key)"
    (let [bs {:internal-states {}
              :variational-free-energy 0.0}]
      (is (not (s/valid? ::wm-spec/belief-state bs)))))

  (testing "Invalid slot (wrong types)"
    (let [slot {:entity-id "not-a-keyword" :position [0] :variance [1]}]
      (is (not (s/valid? ::wm-spec/slot slot))))))

(deftest test-llm-specs
  (testing "Valid candidate-step"
    (let [step {:candidate-action "MOVE"
                :prior-prob 0.8
                :pragmatic-estimate 0.9
                :epistemic-estimate 0.1
                :confidence 1.0}]
      (is (s/valid? ::llm-spec/candidate-step step))))

  (testing "Invalid candidate-step (out of bounds probability)"
    (let [step {:candidate-action "MOVE"
                :prior-prob 1.5 ;; Error: > 1.0
                :pragmatic-estimate 0.9
                :epistemic-estimate 0.1
                :confidence 1.0}]
      (is (not (s/valid? ::llm-spec/candidate-step step)))))

  (testing "Invalid candidate-step (negative confidence)"
    (let [step {:candidate-action "MOVE"
                :prior-prob 0.5
                :pragmatic-estimate 0.5
                :epistemic-estimate 0.5
                :confidence -0.1}] ;; Error: < 0.0
      (is (not (s/valid? ::llm-spec/candidate-step step))))))
