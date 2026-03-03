(ns consider.causal
  "Implementation of Causal Discovery (ALVGL & NOTEARS)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.causal :as causal-spec]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as r]
            [uncomplicate.neanderthal.vect-math :as vm]))

(defn matrix-trace
  "Calculates the trace of a matrix."
  [m]
  (let [d (n/mrows m)]
    (loop [i 0 acc 0.0]
      (if (= i d)
        acc
        (recur (inc i) (+ acc (n/entry m i i)))))))

(defn acyclicity-score
  "Calculates the NOTEARS acyclicity score: h(W) = Tr(exp(W ∘ W)) - d.
   Uses a 4-term Taylor expansion for matrix exponential."
  [W]
  (let [d (n/mrows W)
        ;; W_sq = W ∘ W (Hadamard product)
        W-sq (vm/sqr W)
        I (native/dge d d)]
    (n/scal! 0.0 I)
    (dotimes [i d] (n/entry! I i i 1.0))
    
    (let [M1 W-sq
          M2 (n/mm W-sq W-sq)
          M3 (n/mm M2 W-sq)
          
          ;; exp-M ~ I + M1 + M2/2 + M3/6
          exp-M (n/copy I)]
      (n/axpy! 1.0 M1 exp-M)
      (n/axpy! 0.5 M2 exp-M)
      (n/axpy! (/ 1.0 6.0) M3 exp-M)
      
      (- (matrix-trace exp-M) d))))

(defn alvgl-decomposition
  "Performs a simple Sparse-Low Rank decomposition of a precision matrix."
  [theta lambda gamma]
  ;; This is a dummy implementation of the ALVGL decomposition.
  ;; In a real system, this would use ADMM to solve:
  ;; min ||S||_1 + gamma*||L||_* subject to Theta = S - L
  (let [d (n/mrows theta)
        S (n/copy theta)
        L (native/dge d d)]
    (n/scal! 0.0 L)
    {:sparse-S S
     :low-rank-L L
     :precision-theta theta}))

(defn learn-structure
  "Learns the causal structure from a precision matrix."
  [precision-matrix]
  (alvgl-decomposition precision-matrix 0.1 0.1))
