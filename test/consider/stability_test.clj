(ns consider.stability-test
  (:require [clojure.test :refer :all]
            [consider.core :as core]
            [consider.world-model :as wm]
            [consider.models :as models]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as r]))

(deftest test-long-term-numerical-stability
  (testing "Agent remains stable over 20 steps in a static environment (Anchored baseline)"
    (let [;; Use a high-gain functional vector field that aggressively points to the observation
          v-fn (fn [x t context]
                 (let [obs-v (:observation context)
                       d (n/dim x)
                       v (native/dv d)]
                   (dotimes [i d]
                     (let [target (double (nth obs-v i))
                           current (n/entry x i)]
                       ;; High gain (10.0) to ensure correction exceeds noise
                       (n/entry! v i (* 10.0 (- target current)))))
                   v))

          likelihood (fn [states] [(first (:position (get states :me)))])

          mock-llm (llm/make-mock-llm
                    {[{:role :user :content "Sensory Observation: [1.0]"}]
                     [{:candidate-action "STAY" :prior-prob 1.0 :pragmatic-estimate 1.0 :epistemic-estimate 0.0}]})

          agent (core/initialize-agent {:me (wm/make-slot :me [1.0])}
                                       [[1.0]]
                                       likelihood
                                       v-fn
                                       mock-llm)]

      (loop [curr-agent agent
             step-count 0
             vfe-history []]
        (if (>= step-count 20)
          (do
            (is (every? (complement Double/isNaN) vfe-history) "VFE should never be NaN")
            (is (< (apply max vfe-history) 100.0) "VFE should remain bounded")
            (let [final-pos (first (:position (get-in curr-agent [:belief-state :internal-states :me])))]
              (is (not (Double/isNaN final-pos)) "Internal state should not drift to NaN")
              (is (< (Math/abs (- final-pos 1.0)) 0.2) "Agent should remain stable with high-gain anchored field")))

          (let [res (core/step curr-agent [1.0] {:inference-steps 20 ;; More steps for stronger correction
                                                 :reasoning-iterations 2
                                                 :exploration-weight 1.0})
                vfe (get-in res [:belief-state :variational-free-energy])]
            (recur (assoc curr-agent
                          :belief-state (:belief-state res)
                          :orchestrator-state (:orchestrator-state res)
                          :vector-field-fn (:vector-field-fn res))
                   (inc step-count)
                   (conj vfe-history vfe))))))))
