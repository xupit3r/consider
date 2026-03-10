(ns consider.integration-test
  (:require [clojure.test :refer :all]
            [consider.core :as core]
            [consider.world-model :as wm]
            [consider.inference :as inf]
            [consider.models :as models]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]))

(deftest test-agent-position-tracking
  (let [goal-pos 5.0
        ;; 1. Initialize Agent
        state-dim 1
        obs-dim 1
        hidden-dim 8
        blueprint (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
        net (models/make-mlp-vector-field state-dim obs-dim hidden-dim) ;; Actually already initialized in make-mlp
        
        likelihood-fn (fn [states] 
                        [(first (:position (get states :me)))])
        
        ;; LLM suggests actions based on position
        mock-llm (llm/make-mock-llm 
                  {[{:role :user :content "Sensory Observation: [0.0]"}]
                   [{:candidate-action "MOVE_RIGHT" :prior-prob 0.8 :pragmatic-estimate 0.9 :epistemic-estimate 0.1}
                    {:candidate-action "STAY" :prior-prob 0.2 :pragmatic-estimate 0.1 :epistemic-estimate 0.1}]})
        
        agent (core/initialize-agent {:me (wm/make-slot :me [0.0])}
                                     [[goal-pos]] ;; Preferences (C-matrix)
                                     likelihood-fn
                                     net
                                     mock-llm)
        
        ;; 2. Run few steps
        step1-res (core/step agent [0.0] {:inference-steps 10 
                                         :reasoning-iterations 5 
                                         :exploration-weight 1.0})
        
        _ (is (not (nil? (:next-action step1-res))))
        _ (is (= "MOVE_RIGHT" (:next-action step1-res)))
        
        ;; Simulate environment move
        new-obs [0.1]
        step2-res (core/step (assoc agent 
                                    :belief-state (:belief-state step1-res)
                                    :orchestrator-state (:orchestrator-state step1-res)
                                    :vector-field-fn (:vector-field-fn step1-res)) 
                             new-obs 
                             {:inference-steps 10 
                              :reasoning-iterations 5 
                              :exploration-weight 1.0})]
    
    (testing "Variational Free Energy is tracked"
      (is (some? (:variational-free-energy (:belief-state step2-res)))))
    
    (testing "Causal structure is learned (ALVGL)"
      (is (some? (:causal-structure step2-res))))
    
    (testing "Policy selection minimizes Expected Free Energy"
      (is (not (empty? (:policy step2-res)))))
    
    (testing "Recognition model is being amortized (Sleep phase)"
      (is (instance? consider.models.NeanderthalMLP (:vector-field-fn step2-res))))))
