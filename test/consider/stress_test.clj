(ns consider.stress-test
  (:require [clojure.test :refer :all]
            [consider.core :as core]
            [consider.world-model :as wm]
            [consider.models :as models]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as r]))

;; --- Chaos Mock LLM ---

(defn chaos-predictor
  "A predictor that randomly returns malformed or out-of-bounds data."
  [context]
  (let [r (rand)]
    (cond
      ;; 1. Markdown-wrapped JSON (normal case)
      (< r 0.4) 
      [{:candidate-action "MOVE_FORWARD" :prior-prob 0.8 :pragmatic-estimate 0.9 :epistemic-estimate 0.1}]
      
      ;; 2. Out-of-bounds math components (should be clamped)
      (< r 0.6)
      [{:candidate-action "RISKY_JUMP" :prior-prob 2.5 :pragmatic-estimate -0.5 :epistemic-estimate 0.5}]
      
      ;; 3. Empty candidates (should use defaults)
      (< r 0.8)
      []
      
      ;; 4. Malformed JSON/Garbage (should use defaults)
      :else
      "I am an LLM and I am currently hallucinating garbage: { { [ ] } }")))

(defn chaos-scorer
  "A scorer that periodically returns garbage or missing keys."
  [context action]
  (let [r (rand)]
    (cond
      (< r 0.7)
      {:pragmatic-estimate 0.5 :epistemic-estimate 0.5 :confidence 1.0}
      
      (< r 0.9)
      {:pragmatic-estimate 10.0 :epistemic-estimate -5.0 :confidence 1.0}
      
      :else
      "{\"error\": \"I forgot the schema\"}")))

(defrecord ChaosLLM []
  llm/PolicyPredictor
  (predict-candidates [this context] (chaos-predictor context))
  llm/ProcessScorer
  (score-step [this context action] (chaos-scorer context action)))

(deftest test-long-term-stress-and-dynamic-scaling
  (testing "Agent remains robust under Chaos LLM and repetitive Dynamic Growth (100 Steps)"
    (let [state-dim 1
          obs-dim 1
          hidden-dim 16
          ;; Use a real neural network for growth/amortization testing
          net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)

          flexible-likelihood (fn [states]
                                (mapv #(first (:position %))
                                      (vals (into (sorted-map) states))))

          chaos-llm (->ChaosLLM)
          
          initial-states {:me (wm/make-slot :me [0.0])}
          preferences [[10.0]]
          
          agent (core/initialize-agent initial-states preferences flexible-likelihood net chaos-llm)]

      (loop [curr-agent agent
             step-count 0
             vfe-history []]
        (if (>= step-count 100)
          (do
            (is (every? (complement Double/isNaN) vfe-history) "VFE should never be NaN")
            ;; Large VFE is expected due to Chaos LLM, but it should not be NaN
            (is (< (apply max vfe-history) 1e15) "VFE should remain finite under stress")
            (is (> (count (:internal-states (:belief-state curr-agent))) 1) "Agent should have grown slots")
            (is (instance? consider.models.NeanderthalMLP (:vector-field-fn curr-agent)) "Network should still be valid"))

          (let [;; Periodically introduce novel observations to trigger growth
                ;; Scaled down to prevent massive initial prediction errors
                sensory-data (if (zero? (mod step-count 20))
                               (conj (vec (repeat (count (:internal-states (:belief-state curr-agent))) 0.1))
                                     (double (/ step-count 10.0))) ;; New object appears
                               (vec (repeat (count (:internal-states (:belief-state curr-agent))) 0.1)))
                
                ;; Attempt a step
                res (try
                      (core/step curr-agent sensory-data {:inference-steps 2
                                                          :reasoning-iterations 5
                                                          :exploration-weight 1.0})
                      (catch Exception e
                        (println "CRASH AT STEP" step-count "with observation" sensory-data)
                        (throw e)))
                
                vfe (get-in res [:belief-state :variational-free-energy])]
            
            (recur (assoc curr-agent
                          :belief-state (:belief-state res)
                          :orchestrator-state (:orchestrator-state res)
                          :vector-field-fn (:vector-field-fn res))
                   (inc step-count)
                   (conj vfe-history vfe))))))))
