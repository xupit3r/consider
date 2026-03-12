(ns consider.causal
  "Implementation of Causal Discovery (ALVGL & NOTEARS)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.causal :as causal-spec]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as r]
            [uncomplicate.neanderthal.vect-math :as vm]
            [uncomplicate.neanderthal.linalg :as la]))

(defn soft-threshold
  "Applies the soft-thresholding operator element-wise: sign(x) * max(|x| - tau, 0).
   Forces zero diagonal for causal DAG consistency."
  [m tau]
  (let [res (n/copy m)]
    (dotimes [i (n/mrows res)]
      (dotimes [j (n/ncols res)]
        (if (= i j)
          (n/entry! res i j 0.0)
          (let [v (n/entry res i j)
                abs-v (Math/abs v)
                new-v (* (Math/signum v) (max 0.0 (- abs-v tau)))]
            (n/entry! res i j new-v)))))
    res))

(defn sv-threshold
  "Applies the singular-value thresholding operator: U * ST(Sigma, tau) * Vt.
   Used for nuclear norm regularization (low-rank L)."
  [m tau]
  (let [d (n/mrows m)]
    (if (<= d 1)
      (soft-threshold m tau)
      (try
        (let [svd-res (la/svd (n/copy m) true true)
              u (:u svd-res)
              sigma (:sigma svd-res)
              vt (:vt svd-res)
              d-sig (n/mrows sigma)
              st-sigma-dense (native/dge d-sig d-sig)]
          (n/scal! 0.0 st-sigma-dense)
          (dotimes [i d-sig]
            (let [v (n/entry sigma i i)
                  new-v (max 0.0 (- v tau))]
              (n/entry! st-sigma-dense i i new-v)))
          (n/mm u (n/mm st-sigma-dense vt)))
        (catch Exception _ m)))))

(defn matrix-trace [m]
  (let [d (n/mrows m)]
    (loop [i 0 acc 0.0]
      (if (= i d) acc (recur (inc i) (+ acc (n/entry m i i)))))))

(defn acyclicity-score
  "Calculates the NOTEARS acyclicity score."
  [W]
  (let [d (n/mrows W)
        W-sq (vm/sqr W)
        I (native/dge d d)]
    (n/scal! 0.0 I)
    (dotimes [i d] (n/entry! I i i 1.0))
    (let [M1 W-sq
          M2 (n/mm W-sq W-sq)
          exp-M (n/copy I)]
      (n/axpy! 1.0 M1 exp-M)
      (n/axpy! 0.5 M2 exp-M)
      (- (matrix-trace exp-M) d))))

(defn acyclicity-gradient
  "Calculates the gradient of the NOTEARS acyclicity score."
  [W]
  (let [d (n/mrows W)
        W-sq (vm/sqr W)
        I (native/dge d d)]
    (n/scal! 0.0 I)
    (dotimes [i d] (n/entry! I i i 1.0))
    (let [M1 W-sq
          M2 (n/mm W-sq W-sq)
          exp-M (n/copy I)]
      (n/axpy! 1.0 M1 exp-M)
      (n/axpy! 0.5 M2 exp-M)
      (let [grad (n/trans exp-M)
            two-W (n/copy W)]
        (n/scal! 2.0 two-W)
        (vm/mul! grad two-W)
        (let [gnorm (r/nrm2 grad)]
          (when (> gnorm 1.0) (n/scal! (/ 1.0 gnorm) grad)))
        grad))))

(defn alvgl-decomposition
  "Performs Sparse-Low Rank decomposition of a precision matrix using ADMM with DAG constraints."
  [theta lambda gamma & {:keys [rho beta max-iter tol eta]
                         :or {rho 10.0 beta 0.01 max-iter 100 tol 1e-4 eta 0.01}}]
  (let [d (n/mrows theta)
        S (native/dge d d)
        L (native/dge d d)
        U (native/dge d d)]
    (n/scal! 0.0 S)
    (n/scal! 0.0 L)
    (n/scal! 0.0 U)
    (loop [k 0 S S L L U U]
      (if (>= k max-iter)
        (let [score (acyclicity-score S)]
          (if (> score 1.0)
            {:sparse-S (n/scal! 0.0 (native/dge d d)) :low-rank-L theta :precision-theta theta
             :iterations k :acyclicity 0.0 :warning "Diverged to cyclic structure"}
            {:sparse-S S :low-rank-L L :precision-theta theta :iterations k :acyclicity score}))
        (let [grad (acyclicity-gradient S)
              temp-S (n/copy theta)
              _ (n/axpy! 1.0 L temp-S)
              _ (n/axpy! -1.0 U temp-S)
              _ (n/axpy! (- (* eta beta)) grad temp-S)
              new-S (soft-threshold temp-S (/ lambda rho))
              temp-L (n/copy new-S)
              _ (n/axpy! -1.0 theta temp-L)
              _ (n/axpy! 1.0 U temp-L)
              new-L (sv-threshold temp-L (/ gamma rho))
              new-U (n/copy U)
              _ (n/axpy! 1.0 theta new-U)
              _ (n/axpy! -1.0 new-S new-U)
              _ (n/axpy! 1.0 new-L new-U)
              err (r/nrm2 (n/axpy! -1.0 S (n/copy new-S)))]
          (if (or (< err tol) (Double/isNaN err) (> (r/nrm2 new-S) 1e6))
            (let [score (acyclicity-score new-S)]
              (if (> score 1.0)
                {:sparse-S (n/scal! 0.0 (native/dge d d)) :low-rank-L theta :precision-theta theta
                 :iterations k :acyclicity 0.0 :warning "Early termination due to divergence"}
                {:sparse-S new-S :low-rank-L new-L :precision-theta theta :iterations k :error err :acyclicity score}))
            (recur (inc k) new-S new-L new-U)))))))

(defn learn-structure
  "Learns the causal structure from a precision matrix.
   Normalizes the precision matrix to ensure stable ADMM convergence."
  [precision-matrix]
  (let [d (n/mrows precision-matrix)
        norm-val (r/nrm2 precision-matrix)]
    (if (or (zero? norm-val) (Double/isNaN norm-val))
      (alvgl-decomposition precision-matrix 0.1 0.1)
      (let [normalized-theta (n/copy precision-matrix)]
        (n/scal! (/ 1.0 norm-val) normalized-theta)
        (alvgl-decomposition normalized-theta 0.1 0.1 :rho 50.0)))))
