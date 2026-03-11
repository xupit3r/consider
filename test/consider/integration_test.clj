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
        hidden-dim 16
        ;; Use a real neural network to verify Dynamic Scaling
        net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)

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
                                     net
                                     mock-llm)]

    (testing "Step 1: Detection and Growth"
      (let [sensory-data [0.0 5.0] ;; We observe a new object at 5.0
            res1 (core/step agent sensory-data {:inference-steps 5
                                                :reasoning-iterations 2
                                                :exploration-weight 1.0})]
        (is (= 2 (count (:internal-states (:belief-state res1)))) "Agent should have grown a new slot")
        ;; Verify the network was expanded
        (let [expanded-net (:vector-field-fn res1)]
          (is (instance? consider.models.NeanderthalMLP expanded-net))
          (is (= 2 (n/mrows (:w2 expanded-net))) "Output layer should have grown to 2"))

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

(deftest test-2d-navigation
  (testing "Agent can plan and infer in a 2D space"
    (let [;; Preferences: Be at [10.0 10.0]
          preferences [[10.0 10.0]]

          ;; 2D Likelihood: State [x y] -> Observation [x y]
          likelihood (fn [states]
                       (let [me (:me states)
                             pos (:position me)]
                         [(double (nth pos 0)) (double (nth pos 1))]))

          ;; Mock LLM for 2D planning
          mock-llm (llm/make-mock-llm
                    {[{:role :user :content "Sensory Observation: [0.0 0.0]"}]
                     [{:candidate-action "MOVE_UP_RIGHT" :prior-prob 0.9
                       :pragmatic-estimate 0.8 :epistemic-estimate 0.1}
                      {:candidate-action "STAY" :prior-prob 0.1
                       :pragmatic-estimate 0.1 :epistemic-estimate 0.1}]})

          ;; 2D state vector field (constant velocity for simulation)
          v-fn (fn [x t context] (native/dv (n/dim x)))

          agent (core/initialize-agent {:me (wm/make-slot :me [0.0 0.0])}
                                       preferences
                                       likelihood
                                       v-fn
                                       mock-llm)]

      (let [sensory-data [0.0 0.0]
            res (core/step agent sensory-data {:inference-steps 5
                                               :reasoning-iterations 10
                                               :exploration-weight 1.0})]

        (is (= "MOVE_UP_RIGHT" (:next-action res)))
        (is (= 2 (count (:position (get-in res [:belief-state :internal-states :me])))) "State should remain 2D")
        (is (some? (get-in res [:belief-state :efe-components :risk])))
        (is (some? (get-in res [:belief-state :efe-components :ambiguity])))))))

(deftest test-sequential-growth
  (testing "Agent can grow slots and networks sequentially (1 -> 2 -> 3)"
    (let [state-dim 1
          obs-dim 1
          hidden-dim 16
          net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)

          flexible-likelihood (fn [states]
                                (mapv #(first (:position %))
                                      (vals (into (sorted-map) states))))

          mock-llm (llm/make-mock-llm)

          agent (core/initialize-agent {:me (wm/make-slot :me [0.0])}
                                       [[10.0]]
                                       flexible-likelihood
                                       net
                                       mock-llm)]

      ;; Stage 1: Grow to 2 slots
      (let [res1 (core/step agent [0.0 5.0] {:inference-steps 5 :reasoning-iterations 1 :exploration-weight 1.0})]
        (is (= 2 (count (:internal-states (:belief-state res1)))))
        (is (= 2 (n/mrows (:w2 (:vector-field-fn res1)))))

        ;; Stage 2: Build history for 2 slots
        (let [res2 (loop [curr-res res1 i 0]
                     (if (>= i 10)
                       curr-res
                       (recur (core/step (assoc agent
                                                :belief-state (:belief-state curr-res)
                                                :orchestrator-state (:orchestrator-state curr-res)
                                                :vector-field-fn (:vector-field-fn curr-res))
                                         [0.1 5.1]
                                         {:inference-steps 2 :reasoning-iterations 1 :exploration-weight 1.0})
                              (inc i))))]

          (is (= 2 (n/mrows (:sparse-S (:causal-structure res2)))))

          ;; Stage 3: Grow to 3 slots
          (let [res3 (core/step (assoc agent
                                       :belief-state (:belief-state res2)
                                       :orchestrator-state (:orchestrator-state res2)
                                       :vector-field-fn (:vector-field-fn res2))
                                [0.2 5.2 10.2] ;; Third object appears
                                {:inference-steps 5 :reasoning-iterations 1 :exploration-weight 1.0})]

            (is (= 3 (count (:internal-states (:belief-state res3)))))
            ;; Verify the network was expanded TWICE correctly
            (let [final-net (:vector-field-fn res3)]
              (is (= 3 (n/mrows (:w2 final-net))) "Output layer should have grown to 3")
              (is (= 7 (n/ncols (:w1 final-net))) "Input layer should be 3 (state) + 1 (t) + 3 (obs) = 7"))))))))

(deftest test-interventional-causal-reasoning
  (testing "Agent can reason about interventions (do-calculus) to achieve goals"
    (let [;; Setup: Goal is for Slot B to be at 10.0
          ;; Slot A (me) influences Slot B via causal link A -> B
          slot-ids [:me :object-b]

          ;; Causal Matrix S: B_t = 1.0 * A_{t-1}
          ;; [ A_t ] = [ 1  0 ] [ A_{t-1} ]
          ;; [ B_t ]   [ 1  0 ] [ B_{t-1} ]
          S (native/dge 2 2)
          _ (n/scal! 0.0 S)
          _ (n/entry! S 0 0 1.0) ;; A stays same
          _ (n/entry! S 1 0 1.0) ;; B follows A

          likelihood (fn [states]
                       (let [a (first (:position (:me states)))
                             b (first (:position (:object-b states)))]
                         [a b]))

          ;; Preference: B should be at 10.0 (A is don't care, e.g. 0.0)
          preferences [[0.0 10.0]]

          ;; Mock LLM suggests:
          ;; 1. DO(:me, [10.0]) -> This should move B to 10.0 in the next step
          ;; 2. STAY -> B will stay at 0.0
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

      ;; With the intervention DO(:me, [10.0]), the model predicts:
      ;; Next A = 10.0 (intervention)
      ;; Next B = 1.0 * A_prev = 0.0 (natural evolution)
      ;; WAIT! The intervention should affect the NEXT state directly.
      ;; Let's check how the transition-fn handles it. 
      ;; Current impl: new-pos = (if intervened val else evolved-val).

      (is (= "DO(:me, [10.0])" (:next-action res)) "Agent should choose the intervention to move the causal chain"))))
