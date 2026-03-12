(ns consider.examples-test
  (:require [clojure.test :refer :all]
            [consider.core :as core]
            [consider.world-model :as wm]
            [consider.inference :as inf]
            [consider.models :as models]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.linalg :as la]))

(deftest smart-thermostat-example
  (testing "Adaptive Climate Control: Preference Matching"
    (let [target-temp 22.0
          initial-states {:room (wm/make-slot :room [18.0 0.0])}
          preferences [[target-temp 0.0]]
          likelihood-fn (fn [states]
                          (let [room (:room states)
                                pos (:position room)]
                            [(double (nth pos 0)) (double (nth pos 1))]))
          v-fn (fn [x t ctx] (native/dv (n/dim x)))
          mock-llm (llm/make-mock-llm
                    {[{:role :user :content "Sensory Observation: [18.0 0.0]"}]
                     [{:candidate-action "TURN_ON_HEATER" :prior-prob 0.8
                       :pragmatic-estimate 0.9 :epistemic-estimate 0.1}
                      {:candidate-action "STAY" :prior-prob 0.2
                       :pragmatic-estimate 0.1 :epistemic-estimate 0.1}]})

          agent (core/initialize-agent initial-states preferences likelihood-fn v-fn mock-llm)]

      (let [res (core/step agent [18.0 0.0] {:inference-steps 5
                                             :reasoning-iterations 10
                                             :exploration-weight 1.0})]

        (is (= "TURN_ON_HEATER" (:next-action res)) "Agent should decide to heat up to reach preference")))))

(deftest novelty-discovery-example
  (testing "Novelty Discovery: Dynamic Slot Spawning"
    (let [initial-states {:me (wm/make-slot :me [0.0])}
          preferences [[10.0]]
          ;; Likelihood that handles dynamic number of slots
          flexible-likelihood (fn [states]
                                (mapv #(first (:position %))
                                      (vals (into (sorted-map) states))))

          v-fn (fn [x t ctx] (native/dv (n/dim x)))
          mock-llm (llm/make-mock-llm)

          agent (core/initialize-agent initial-states preferences flexible-likelihood v-fn mock-llm)]

      (testing "Step 1: Normal state"
        (let [res1 (core/step agent [0.0] {:inference-steps 1 :reasoning-iterations 1 :exploration-weight 1.0})]
          (is (= 1 (count (:internal-states (:belief-state res1)))))))

      (testing "Step 2: Novel entity appears"
        (let [res2 (core/step agent [0.0 5.0] {:inference-steps 5 :reasoning-iterations 1 :exploration-weight 1.0})]
          (is (= 2 (count (:internal-states (:belief-state res2)))) "Agent should have grown a new slot for the novel observation")
          (is (some? (filter (fn [[id _]] (clojure.string/includes? (name id) "entity-"))
                             (:internal-states (:belief-state res2))))))))))

(deftest collaborative-assistant-example
  (testing "Social Navigation: LLM-based State Scoring"
    (let [;; Setup: Agent assisting with a 'Research Task'
          ;; States: [UserProgress, UserFrustration]
          initial-states {:interaction (wm/make-slot :interaction [0.0 0.0])}

          ;; Goal: Max Progress (1.0), Min Frustration (0.0)
          preferences [[1.0 0.0]]

          likelihood-fn (fn [states] (vec (:position (:interaction states))))

          ;; Scorer Mock: LLM evaluates the 'Research Context'
          ;; A helpful explanation reduces frustration and increases progress scent.
          mock-llm (llm/make-mock-llm
                    {[{:role :user :content "Sensory Observation: [0.0 0.0]"}]
                     [{:candidate-action "EXPLAIN_CONCEPT" :prior-prob 0.7
                       :pragmatic-estimate 0.8 :epistemic-estimate 0.4}
                      {:candidate-action "ASK_FOR_FEEDBACK" :prior-prob 0.3
                       :pragmatic-estimate 0.3 :epistemic-estimate 0.9}]})

          agent (core/initialize-agent initial-states preferences likelihood-fn (fn [x t ctx] (native/dv (n/dim x))) mock-llm)]

      (let [res (core/step agent [0.0 0.0] {:inference-steps 1
                                            :reasoning-iterations 20
                                            :exploration-weight 1.0})]

        ;; EXPLAIN_CONCEPT G = (1 - 0.8) - 0.4 = -0.2
        ;; ASK_FOR_FEEDBACK G = (1 - 0.3) - 0.9 = -0.2
        ;; Both are good, but EXPLAIN_CONCEPT has higher prior.

        (is (contains? #{"EXPLAIN_CONCEPT" "ASK_FOR_FEEDBACK"} (:next-action res)))
        (is (some? (get-in res [:belief-state :efe-components :risk])))
        (is (some? (get-in res [:belief-state :efe-components :ambiguity])))))))

(deftest curious-explorer-example
  (testing "Epistemic Value: Resolving Ambiguity via Active Sensing"
    (let [;; Preferences: Be at the Reward (100.0)
          preferences [[100.0]]

          ;; Likelihood: Goal is invisible (0.0) unless light is ON (dim 1 = 1.0)
          likelihood (fn [states]
                       (let [pos (:position (:me states))
                             goal-pos 100.0
                             light-on? (> (nth pos 1) 0.5)]
                         [(if light-on? goal-pos 0.0)]))

          ;; Mock LLM:
          ;; 1. TURN_ON_LIGHT: No immediate progress (pragmatic 0.0), high info gain (epistemic 1.0)
          ;; 2. WALK_BLINDLY: Random progress (pragmatic 0.1), low info gain (epistemic 0.0)
          mock-llm (llm/make-mock-llm
                    {[{:role :user :content "Sensory Observation: [0.0]"}]
                     [{:candidate-action "TURN_ON_LIGHT" :prior-prob 0.4
                       :pragmatic-estimate 0.1 :epistemic-estimate 1.0}
                      {:candidate-action "WALK_BLINDLY" :prior-prob 0.6
                       :pragmatic-estimate 0.2 :epistemic-estimate 0.1}]})

          agent (core/initialize-agent {:me (wm/make-slot :me [0.0 0.0])}
                                       preferences
                                       likelihood
                                       (fn [x t ctx] (native/dv (n/dim x)))
                                       mock-llm)

          res (core/step agent [0.0] {:inference-steps 1
                                      :reasoning-iterations 20
                                      :exploration-weight 1.0})]

      ;; TURN_ON_LIGHT G = (1 - 0.1) - 1.0 = -0.1
      ;; WALK_BLINDLY G   = (1 - 0.2) - 0.1 = 0.7
      ;; Agent should choose TURN_ON_LIGHT despite lower prior and slightly lower pragmatic

      (is (= "TURN_ON_LIGHT" (:next-action res)) "Agent should seek information to resolve goal ambiguity"))))
