(ns consider.specs.causal
  "Specifications for the Differentiable Causal Discovery Engine (ALVGL)."
  (:require [clojure.spec.alpha :as s]))

;; --- Matrix Components (Neanderthal/Native) ---
;; These represent off-heap DirectByteBuffers managed by Neanderthal.
(s/def ::neanderthal-matrix any?) ;; uncomplicate.neanderthal.core/Matrix
(s/def ::neanderthal-vector any?) ;; uncomplicate.neanderthal.core/Vector

;; --- ALVGL Decomposition: Θ = S - L ---
;; S: Sparse component representing direct causal links (Adjacency).
(s/def ::sparse-S ::neanderthal-matrix)

;; L: Low-rank component representing latent confounders.
(s/def ::low-rank-L ::neanderthal-matrix)

;; Θ: Precision matrix (inverse covariance).
(s/def ::precision-theta ::neanderthal-matrix)

(s/def ::alvgl-decomposition
  (s/keys :req-un [::sparse-S ::low-rank-L ::precision-theta]))

;; --- ADMM Optimization Variables ---
;; Alternating Direction Method of Multipliers state.
(s/def ::dual-variables ::neanderthal-matrix) ;; Lagrange multipliers (U)
(s/def ::rho float?) ;; Penalty parameter for ADMM
(s/def ::lambda float?) ;; L1 penalty for sparsity of S
(s/def ::gamma float?) ;; Nuclear norm penalty for low-rank L

(s/def ::admm-state
  (s/keys :req-un [::sparse-S ::low-rank-L ::dual-variables ::rho ::lambda ::gamma]))

;; --- Continuous Acyclicity (NOTEARS) ---
;; h(W) = Tr(exp(W ∘ W)) - d = 0
(s/def ::acyclicity-score float?) ;; Value of h(W)
(s/def ::d-dimension pos-int?) ;; Number of variables (nodes)

(s/def ::dag-constraint
  (s/keys :req-un [::acyclicity-score ::d-dimension]))

;; --- Causal Discovery Interface ---
(s/def ::learn-structure-fn fn?) ;; (fn [precision-matrix] -> alvgl-decomposition)
