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
      (is (sequential? (:policy result))))))
