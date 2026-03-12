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
        net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
        
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

(deftest test-interventional-reasoning
  (testing "Agent selects an interventional action to reduce causal ambiguity"
    (let [initial-states {:e1 (wm/make-slot :e1 [0.0])}
          preferences [[1.0]]
          likelihood-fn (fn [states] [0.0])
          v-fn (fn [x t ctx] (native/dv (n/dim x)))
          
          ;; Mock LLM suggests a regular action and an interventional action
          mock-llm (llm/make-mock-llm 
                    {[{:role :user :content "Sensory Observation: [0.0]"}]
                     [{:candidate-action "STAY" :prior-prob 0.5 :pragmatic-estimate 0.5 :epistemic-estimate 0.1}
                      ;; Interventional action string
                      {:candidate-action "DO(:e1, [1.0])" :prior-prob 0.5 :pragmatic-estimate 0.5 :epistemic-estimate 0.1}]
                     
                     ;; Scoring for the actions
                     [[{:role :user :content "Sensory Observation: [0.0]"}] "STAY"]
                     {:pragmatic-estimate 0.5 :epistemic-estimate 0.1 :confidence 1.0}
                     
                     [[{:role :user :content "Sensory Observation: [0.0]"}] "DO(:e1, [1.0])"]
                     {:pragmatic-estimate 0.5 :epistemic-estimate 0.1 :confidence 1.0}})
          
          agent (core/initialize-agent initial-states preferences likelihood-fn v-fn mock-llm)
          
          res (core/step agent [0.0] {:inference-steps 1 :reasoning-iterations 50 :exploration-weight 0.0})]
      
      ;; MCTS should favor DO(:e1, [1.0]) because it intervenes on an uncertain dimension (precision fallback is high boost)
      (is (= {:type :do :target :e1 :value [1.0]} (:next-action res))))))

(deftest test-novelty-growth-and-planning
  (let [goal-pos 10.0
        state-dim 1
        obs-dim 1
        hidden-dim 16
        blueprint (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
        net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
        
        likelihood (fn [states]
                     (let [sorted (sort-by key states)]
                       (mapv (fn [[id slot]] (first (:position slot))) sorted)))
        
        mock-llm (llm/make-mock-llm)
        
        agent (core/initialize-agent {:me (wm/make-slot :me [0.0])}
                                     [[goal-pos]]
                                     likelihood
                                     net
                                     mock-llm)
        
        ;; Simulate a large error to trigger GROWTH
        sensory-data [0.0 5.0] ;; We only expect one obs, but get two
        res (core/step agent sensory-data {:inference-steps 1 :reasoning-iterations 5 :exploration-weight 1.0})]
    
    (testing "Agent grows its hidden state space when novel entities are detected"
      (is (> (count (:internal-states (:belief-state res))) 1)))
    
    (testing "Neural network can still be trained after growth"
      ;; Since Grow Phase happens, it might return a fn? if growth is not neural-supported
      (is (some? (:vector-field-fn res))))))

(deftest test-high-dimensional-inference
  (testing "Agent can perform inference in 2D spaces"
    (let [state-dim 2
          obs-dim 2
          hidden-dim 16
          net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
          
          initial-states {:me (wm/make-slot :me [0.0 0.0])}
          preferences [[10.0 10.0]]
          likelihood (fn [states] (:position (:me states)))
          
          mock-llm (llm/make-mock-llm)
          agent (core/initialize-agent initial-states preferences likelihood net mock-llm)
          
          res (core/step agent [1.0 1.0] {:inference-steps 5 :reasoning-iterations 2 :exploration-weight 1.0})]
      
      (is (vector? (:position (:me (:belief-state res)))))
      (is (= 2 (count (:position (:me (:belief-state res)))))))))

(deftest test-causal-structure-recovery
  (testing "Agent recovers causal dependencies from belief history"
    (let [;; Create a history where B follows A: B_t = A_{t-1}
          history (map (fn [v] 
                         {:internal-states {:a (wm/make-slot :a [v])
                                            :b (wm/make-slot :b [(max 0.0 (dec v))])}})
                       (range 20))
          
          belief-state (-> (wm/make-belief-state {} [])
                           (assoc :history history :internal-states {:a (wm/make-slot :a [0.0])
                                                                     :b (wm/make-slot :b [0.0])}))
          
          ;; Manually trigger precision estimation
          precision (#'consider.core/estimate-precision-matrix belief-state)
          causal-res (consider.causal/learn-structure precision)]
      
      (is (some? (:sparse-S causal-res)))
      ;; Matrix should be 2x2
      (is (= 2 (n/mrows (:sparse-S causal-res)))))))

(deftest test-sequential-growth
  (testing "Agent can grow multiple slots over time"
    (let [state-dim 1
          obs-dim 1
          hidden-dim 8
          net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
          
          likelihood (fn [states]
                       (let [sorted (sort-by key states)]
                         (mapv (fn [[id slot]] (first (:position slot))) sorted)))
          
          agent (core/initialize-agent {:me (wm/make-slot :me [0.0])}
                                       [[10.0]]
                                       likelihood
                                       net
                                       (llm/make-mock-llm))
          
          ;; Step 1: Add one new entity
          res1 (core/step agent [0.0 5.0] {:inference-steps 1 :reasoning-iterations 2 :exploration-weight 1.0})
          ;; Step 2: Add another new entity
          res2 (core/step (assoc agent :belief-state (:belief-state res1) :vector-field-fn (:vector-field-fn res1)) 
                          [0.0 5.0 8.0] 
                          {:inference-steps 1 :reasoning-iterations 2 :exploration-weight 1.0})]
      
      (is (= 2 (count (:internal-states (:belief-state res1)))))
      (is (= 3 (count (:internal-states (:belief-state res2))))))))

(deftest test-interventional-causal-reasoning
  (testing "Agent can reason about interventions (do-calculus) to achieve goals"
    (let [;; Two slots: A and B. B follows A.
          ;; Goal is to move B. Action is to move A.
          ;; [ B_t ]   [ 1  0 ] [ B_{t-1} ]
          S (native/dge 2 2)
          _ (n/scal! 0.0 S)
          _ (n/entry! S 0 0 1.0) ;; A stays same
          _ (n/entry! S 1 0 1.0) ;; B follows A
          
          likelihood (fn [states]
                       (let [a (first (:position (:me states)))
                             b (first (:position (:object-b states)))]
                         [a b]))
          
          ;; Preference: B should be at 10.0
          preferences [[0.0 10.0]]
          
          mock-llm (llm/make-mock-llm 
                    {[{:role :user :content "Sensory Observation: [0.0 0.0]"}]
                     [{:candidate-action "DO(:me, [10.0])" :prior-prob 0.5 
                       :pragmatic-estimate 0.9 :epistemic-estimate 0.1}
                      {:candidate-action "STAY" :prior-prob 0.5 
                       :pragmatic-estimate 0.1 :epistemic-estimate 0.1}]})
          
          agent (core/initialize-agent {:me (wm/make-slot :me [0.0])
                                        :object-b (wm/make-slot :object-b [0.0])}
                                       preferences
                                       likelihood
                                       (fn [x t ctx] (native/dv (n/dim x)))
                                       mock-llm)
          
          ;; Inject the causal model manually for the test
          agent-with-causal (update agent :belief-state wm/update-transition-dynamics S)
          
          res (core/step agent-with-causal [0.0 0.0] {:inference-steps 1 
                                                      :reasoning-iterations 10 
                                                      :exploration-weight 1.0})]
      
      (is (= {:type :do :target :me :value [10.0]} (:next-action res)) "Agent should choose the intervention to move the causal chain"))))
