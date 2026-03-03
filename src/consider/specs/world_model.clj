(ns consider.specs.world-model
  "Specifications for the Probabilistic World Model (Active Inference)."
  (:require [clojure.spec.alpha :as s]))

;; --- Markov Blanket Components ---

;; Internal States (μ): The agent's belief about the hidden causes of its sensations.
(s/def ::entity-id keyword?)
(s/def ::position (s/coll-of float? :kind vector?))
(s/def ::variance (s/coll-of float? :kind vector?))

;; A 'Slot' represents a hidden state (s) in the generative model.
(s/def ::slot (s/keys :req-un [::entity-id ::position ::variance]))
(s/def ::internal-states (s/map-of ::entity-id ::slot))

;; Sensory States (o): Observations received from the external environment.
(s/def ::observation (s/map-of keyword? any?)) 
(s/def ::sensory-states (s/coll-of ::observation :kind vector?))

;; Active States (a): Actions or policies selected by the internal states to perturb the environment.
(s/def ::action-vector (s/coll-of float? :kind vector?))
(s/def ::policy (s/coll-of ::action-vector :kind vector?)) ;; A sequence of actions

;; --- Generative Model Components ---

;; Likelihood P(o|s): Mapping from hidden states to predicted observations.
(s/def ::likelihood-mapping fn?) ;; (fn [internal-states] -> predicted-sensory-states)

;; Transition Dynamics P(s_t+1 | s_t, a_t): Mapping from current states and action to next states.
(s/def ::transition-dynamics fn?) ;; (fn [internal-states action] -> predicted-internal-states)

;; Priors P(s): Beliefs about states before observation, including preferences (C-matrix).
(s/def ::prior-beliefs (s/keys :req-un [::internal-states]))
(s/def ::preferences (s/coll-of ::observation :kind vector?)) ;; Goal states (C)

;; --- Variational & Expected Free Energy ---

(s/def ::variational-free-energy float?) ;; F = Divergence + Complexity (or Complexity - Accuracy)
(s/def ::expected-free-energy float?)    ;; G = Risk + Ambiguity

;; Decomposition of G (Expected Free Energy)
(s/def ::risk (s/nilable float?))        ;; Pragmatic value (KL divergence from preferences)
(s/def ::ambiguity (s/nilable float?))   ;; Epistemic value (Information gain/Exploration)

(s/def ::efe-components (s/keys :req-un [::risk ::ambiguity]))

;; --- Global State ---

(s/def ::belief-state
  (s/keys :req-un [::internal-states 
                 ::variational-free-energy 
                 ::expected-free-energy 
                 ::efe-components
                 ::preferences]
          :opt-un [::likelihood-mapping
                 ::transition-dynamics]))
