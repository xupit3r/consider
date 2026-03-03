(ns consider.llm
  "Implementation of the LLM Abstraction Layer (System 1)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.llm :as llm-spec]))

;; --- Protocol definitions (re-stated from specs for actual use) ---

(defprotocol PolicyPredictor
  "Protocol for generating candidate reasoning steps and their priors."
  (predict-candidates [this context] "Returns a set of candidate-steps."))

(defprotocol ProcessScorer
  "Protocol for scoring a candidate step or path (Process Reward)."
  (score-step [this context candidate-action] "Returns a pragmatic and epistemic estimate."))

;; --- Mock Implementation for Testing ---

(defrecord MockLLM [responses]
  PolicyPredictor
  (predict-candidates [this context]
    (or (get responses context)
        [{:candidate-action "Standard Reasoning Step"
          :prior-prob 1.0
          :pragmatic-estimate 0.5
          :epistemic-estimate 0.5
          :confidence 1.0}]))
  
  ProcessScorer
  (score-step [this context candidate-action]
    {:pragmatic-estimate 0.5
     :epistemic-estimate 0.5
     :confidence 1.0}))

(defn make-mock-llm
  "Creates a mock LLM with predefined responses."
  ([] (make-mock-llm {}))
  ([responses] (->MockLLM responses)))

(defn validate-candidate-step
  "Validates a candidate step against its specification."
  [step]
  (if (s/valid? ::llm-spec/candidate-step step)
    step
    (throw (ex-info "Invalid candidate step"
                    {:explain (s/explain-data ::llm-spec/candidate-step step)}))))
