(ns consider.specs.executive
  "Specifications for the Executive Orchestrator (Policy Selection via MCTS & FoT)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.llm :as llm]
            [consider.specs.world-model :as wm]))

;; --- Policy Selection (Active Inference) ---
;; The executive selects the policy (π) that minimizes Expected Free Energy (G).

(s/def ::node-id string?)
(s/def ::parent-id (s/nilable ::node-id))

;; State represents the partial trajectory of internal beliefs and observations.
(s/def ::state (s/coll-of ::llm/message :vec true))

;; Action (u) is a discrete reasoning or physical step in the environment.
(s/def ::action string?)

;; --- MCTS Node Values (EFE Decomposition) ---
;; Q-value represents the Expected Free Energy (G) for a policy path.
(s/def ::value float?)       ;; Total G(π) = Risk + Ambiguity
(s/def ::risk float?)        ;; Pragmatic value (derived from preferences)
(s/def ::ambiguity float?)   ;; Epistemic value (information gain)

;; Visits (N) and Policy Prior (P)
;; P is derived from the LLM "System 1" as a prior over policies.
(s/def ::visits int?)
(s/def ::prior-prob float?)  ;; P(π) prior probability from System 1

;; A node in a reasoning tree representing a potential future belief state.
(s/def ::node
  (s/keys :req-un [::node-id ::parent-id ::state ::action 
                 ::value ::risk ::ambiguity 
                 ::visits ::prior-prob]))

;; --- Forest-of-Thought (FoT) ---
;; A collection of reasoning trees exploring different policy horizons.
(s/def ::tree (s/coll-of ::node :kind map?)) ;; map of node-id to node
(s/def ::forest (s/map-of keyword? ::tree)) ;; map of tree-id to tree

;; --- Orchestrator & Sparse Activation ---
;; Pruning branches that lead to high EFE (low value/reward).
(s/def ::max-depth pos-int?)
(s/def ::max-compute-tokens pos-int?)
(s/def ::active-branches (s/coll-of ::node-id :set true))

(s/def ::orchestrator-state
  (s/keys :req-un [::forest ::active-branches ::max-depth ::max-compute-tokens]))

;; --- Executive Protocols ---
(s/def ::policy-selection-fn fn?) ;; (fn [belief-state] -> optimal-policy)
