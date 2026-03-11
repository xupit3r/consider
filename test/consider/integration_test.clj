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

(deftest test-novelty-growth-and-planning
  (let [goal-pos 10.0
        state-dim 1
        obs-dim 1
        hidden-dim 8
        ;; Use a functional vector field that scales with slots
        v-fn (fn [x t context]
               (let [d (n/dim x)
                     v (native/dv d)]
                 (n/scal! 0.0 v) ;; Constant velocity for simulation
                 v))

        ;; A likelihood function that adapts to the number of slots
        flexible-likelihood (fn [states]
                              (mapv #(first (:position %))
                                    (vals (into (sorted-map) states))))

        ;; Mock LLM for planning
        mock-llm (llm/make-mock-llm
                  {[{:role :user :content "Sensory Observation: [0.0 5.0]"}]
                   [{:candidate-action "MOVE_RIGHT" :prior-prob 0.9 :pragmatic-estimate 0.8 :epistemic-estimate 0.2}
                    {:candidate-action "STAY" :prior-prob 0.1 :pragmatic-estimate 0.2 :epistemic-estimate 0.1}]})

        agent (core/initialize-agent {:me (wm/make-slot :me [0.0])}
                                     [[goal-pos]] ;; Preferences
                                     flexible-likelihood
                                     v-fn
                                     mock-llm)]

    (testing "Step 1: Detection and Growth"
      (let [sensory-data [0.0 5.0] ;; We observe a new object at 5.0
            res1 (core/step agent sensory-data {:inference-steps 5
                                                :reasoning-iterations 2
                                                :exploration-weight 1.0})]
        (is (= 2 (count (:internal-states (:belief-state res1)))) "Agent should have grown a new slot")

        (testing "Step 2 & 3: Historical Causal Learning"
          (let [res2 (core/step (assoc agent
                                       :belief-state (:belief-state res1)
                                       :orchestrator-state (:orchestrator-state res1)
                                       :vector-field-fn (:vector-field-fn res1))
                                [0.1 5.1]
                                {:inference-steps 5 :reasoning-iterations 2 :exploration-weight 1.0})
                res3 (core/step (assoc agent
                                       :belief-state (:belief-state res2)
                                       :orchestrator-state (:orchestrator-state res2)
                                       :vector-field-fn (:vector-field-fn res2))
                                [0.2 5.2]
                                {:inference-steps 5 :reasoning-iterations 2 :exploration-weight 1.0})]

            (is (= 3 (count (:history (:belief-state res3)))) "History should be maintained")

            (testing "Empirical Causal Discovery"
              (let [causal (:causal-structure res3)
                    sparse-S (:sparse-S causal)]
                (is (some? causal))
                (is (= 2 (n/mrows sparse-S)))
                (is (some? (:acyclicity causal)))))))))

    (testing "Step 4: Planning with Refined EFE"
      (let [sensory-data [0.3 5.3]
            res4 (core/step agent sensory-data {:inference-steps 5
                                                :reasoning-iterations 5
                                                :exploration-weight 1.0})]
        (is (not (empty? (:policy res4))))
        (is (some? (get-in res4 [:belief-state :efe-components :risk])))
        (is (some? (get-in res4 [:belief-state :efe-components :ambiguity])))))))

(deftest test-active-foraging
  (testing "Agent prioritizes information-seeking (Epistemic Value) over risky exploitation"
    (let [;; Likelihood: deterministic mapping from position to observation
          ;; If position is 1.0 (Hint), reveal Goal at 10.0. Otherwise Goal is hidden (0.0).
          likelihood (fn [states]
                       (let [me-pos (first (:position (get states :me)))
                             goal-target 10.0]
                         [me-pos (if (> me-pos 0.9) goal-target 0.0)]))

          ;; Preferences: Be at the Goal Location (10.0)
          preferences [[10.0 10.0]]

          ;; Mock LLM:
          ;; 1. MOVE_TO_HINT: low pragmatic (far from goal), high epistemic (it's a hint)
          ;; 2. GUESS_GOAL: medium pragmatic (might be goal), low epistemic
          mock-llm (llm/make-mock-llm
                    {[{:role :user :content "Sensory Observation: [0.0 0.0]"}]
                     [{:candidate-action "MOVE_TO_HINT" :prior-prob 0.5
                       :pragmatic-estimate 0.1 :epistemic-estimate 0.9}
                      {:candidate-action "GUESS_GOAL" :prior-prob 0.5
                       :pragmatic-estimate 0.5 :epistemic-estimate 0.1}]})

          agent (core/initialize-agent {:me (wm/make-slot :me [0.0])}
                                       preferences
                                       likelihood
                                       (fn [x t ctx] (native/dv (n/dim x)))
                                       mock-llm)

          ;; Run reasoning
          res (core/step agent [0.0 0.0] {:inference-steps 1
                                          :reasoning-iterations 10
                                          :exploration-weight 1.0})]

      ;; MOVE_TO_HINT G = (1 - 0.1) - 0.9 = 0.0
      ;; GUESS_GOAL G   = (1 - 0.5) - 0.1 = 0.4
      ;; Orchestrator should pick MOVE_TO_HINT (lower G)
      (is (= "MOVE_TO_HINT" (:next-action res)) "Agent should forage for the hint first"))))
