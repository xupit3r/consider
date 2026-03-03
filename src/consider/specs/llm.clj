(ns consider.specs.llm
  "Specifications for the LLM Abstraction Layer (System 1 Policy & Scorer)."
  (:require [clojure.spec.alpha :as s]))

;; --- System 1: Fast Policy Prior (P) ---

(s/def ::role #{:system :user :assistant :tool})
(s/def ::content string?)
(s/def ::message (s/keys :req-un [::role ::content]))
(s/def ::context (s/coll-of ::message :vec true))

;; The completion string is treated as a candidate Action (u) in the MCTS.
(s/def ::candidate-action string?)

;; Policy Prior: P(π) for MCTS selection.
(s/def ::prior-prob (s/and float? #(<= 0.0 % 1.0)))

;; --- Process Reward Model (Scorer) ---

;; Heuristic estimate of the pragmatic value (goal relevance).
(s/def ::pragmatic-estimate float?)

;; Heuristic estimate of the epistemic value (information/reasoning quality).
(s/def ::epistemic-estimate float?)

;; Confidence: Degree of certainty in the heuristic estimates.
(s/def ::confidence (s/and float? #(<= 0.0 % 1.0)))

(s/def ::candidate-step
  (s/keys :req-un [::candidate-action ::prior-prob 
                 ::pragmatic-estimate ::epistemic-estimate 
                 ::confidence]))

;; System 1 generates a set of candidates for MCTS Expansion.
(s/def ::candidates (s/coll-of ::candidate-step :min-count 1))

;; --- LLM Protocols ---

(defprotocol PolicyPredictor
  "Protocol for generating candidate reasoning steps and their priors."
  (predict-candidates [this context] "Returns a set of candidate-steps."))

(defprotocol ProcessScorer
  "Protocol for scoring a candidate step or path (Process Reward)."
  (score-step [this context candidate-action] "Returns a pragmatic and epistemic estimate."))
