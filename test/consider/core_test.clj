(ns consider.core-test
  (:require [clojure.test :refer :all]
            [consider.core :refer :all]
            [consider.world-model :as wm]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]))

(deftest test-agent-step-integration
  (let [mock-llm (llm/make-mock-llm)
        likelihood-fn (fn [states] [0.0])
        vector-field-fn (fn [x t context] (native/dv (n/dim x)))
        agent (initialize-agent {:e1 (wm/make-slot :e1 [0.0])} 
                                [] ;; Preferences
                                likelihood-fn 
                                vector-field-fn 
                                mock-llm)
        
        result (step agent [0.1] {:inference-steps 5 
                                  :reasoning-iterations 3 
                                  :exploration-weight 1.0})]
    
    (testing "Belief update happened"
      (is (contains? (:belief-state result) :variational-free-energy)))
    
    (testing "Causal structure was learned"
      (is (contains? (:causal-structure result) :sparse-S)))
    
    (testing "Decision was made"
      (is (not (nil? (:next-action result))))
      (is (sequential? (:policy result))))
    
    (testing "Transition dynamics closed-loop update"
      (let [updated-belief (:belief-state result)]
        (is (some? (:transition-dynamics updated-belief)))
        ;; Verify that transition-dynamics is a function (fn [internal-states action])
        (is (ifn? (:transition-dynamics updated-belief)))
        ;; Test the function
        (let [next-states ((:transition-dynamics updated-belief) 
                           (:internal-states updated-belief) 
                           "test-action")]
          (is (map? next-states))
          (is (contains? next-states :e1))
          (is (vector? (:position (get next-states :e1)))))))))
