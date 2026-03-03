(ns consider.specs.inference
  "Specifications for Next-Generation Amortized Variational Inference (FlowNP & DDVI)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.world-model :as wm]))

;; --- Recognition Density q(s) ---
;; The internal distribution parameterized by neural networks (FlowNP/DDVI)
(s/def ::recognition-density (s/keys :req-un [::wm/internal-states]))

;; Flow Matching (Continuous Normalizing Flows)
;; Mapping a simple prior (standard normal) to the complex posterior (belief state).
(s/def ::time float?) ;; t in [0, 1] representing the flow from noise to signal.
(s/def ::state-vector (s/coll-of float? :kind vector?))
(s/def ::vector-field fn?) ;; (fn [t state-vector] -> velocity)

;; Context-dependent Belief Updating
;; FlowNP uses context (sensory data) to predict the flow field parameters.
(s/def ::context-sensory-data ::wm/sensory-states)
(s/def ::context-points (s/coll-of (s/tuple ::state-vector ::state-vector)))

;; --- Variational Objectives ---

;; Evidence Lower Bound (ELBO) Optimization
;; Wake-Sleep ELBO used in DDVI to align the generative model p(o,s) and recognition model q(s|o).
(s/def ::elbo float?) 
(s/def ::complexity float?) ;; KL(q(s) || p(s))
(s/def ::accuracy float?)   ;; E_q[ln p(o|s)]

(s/def ::variational-metrics (s/keys :req-un [::elbo ::complexity ::accuracy]))

;; --- Inference Engine Interface ---

(s/def ::belief-update-fn fn?) ;; (fn [prior-belief sensory-input] -> updated-belief)
