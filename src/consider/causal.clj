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
  "Applies the soft-thresholding operator element-wise: sign(x) * max(|x| - tau, 0)."
  [m tau]
  (let [res (n/copy m)]
    (dotimes [i (n/mrows res)]
      (dotimes [j (n/ncols res)]
        (let [v (n/entry res i j)
              abs-v (Math/abs v)
              new-v (* (Math/signum v) (max 0.0 (- abs-v tau)))]
          (n/entry! res i j new-v))))
    res))

(defn sv-threshold
  "Applies the singular-value thresholding operator: U * ST(Sigma, tau) * Vt."
  [m tau]
  (let [svd-res (la/svd m true true)
        u (:u svd-res)
        sigma (:sigma svd-res)
        vt (:vt svd-res)
        ;; Sigma is a diagonal matrix or a vector.
        is-vctr (n/vctr? sigma)
        d (if is-vctr (n/dim sigma) (n/mrows sigma))
        st-sigma (n/copy sigma)]
    (dotimes [i d]
      ;; Use entry with two indices for diagonal matrices, or one for vectors.
      (let [v (if is-vctr (n/entry st-sigma i) (n/entry st-sigma i i))
            new-v (max 0.0 (- v tau))]
        (if is-vctr
          (n/entry! st-sigma i new-v)
          (n/entry! st-sigma i i new-v))))
    ;; Reconstruct: u * st-sigma * vt
    (let [st-sigma-dense (if is-vctr
                           (let [diag-m (native/dge d d)]
                             (n/scal! 0.0 diag-m)
                             (dotimes [i d] (n/entry! diag-m i i (n/entry st-sigma i)))
                             diag-m)
                           st-sigma)]
      (n/mm u (n/mm st-sigma-dense vt)))))

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

(defn acyclicity-gradient
  "Calculates the gradient of the NOTEARS acyclicity score: ∇h(W) = exp(W ∘ W)^T ∘ 2W."
  [W]
  (let [d (n/mrows W)
        W-sq (vm/sqr W)
        I (native/dge d d)]
    (n/scal! 0.0 I)
    (dotimes [i d] (n/entry! I i i 1.0))
    
    (let [M1 W-sq
          M2 (n/mm W-sq W-sq)
          M3 (n/mm M2 W-sq)
          exp-M (n/copy I)]
      (n/axpy! 1.0 M1 exp-M)
      (n/axpy! 0.5 M2 exp-M)
      (n/axpy! (/ 1.0 6.0) M3 exp-M)
      
      (let [grad (n/trans exp-M)
            two-W (n/copy W)]
        (n/scal! 2.0 two-W)
        ;; Hadamard product: grad ∘ 2W
        (vm/mul! grad two-W)))))

(defn alvgl-decomposition
  "Performs Sparse-Low Rank decomposition of a precision matrix using ADMM with DAG constraints.
   Solves: min ||S||_1 + gamma*||L||_* + beta*h(S) subject to Theta = S - L."
  [theta lambda gamma & {:keys [rho beta max-iter tol eta] 
                         :or {rho 1.0 beta 0.5 max-iter 100 tol 1e-4 eta 0.01}}]
  (let [d (n/mrows theta)
        S (native/dge d d)
        L (native/dge d d)
        U (native/dge d d)]
    (n/scal! 0.0 S)
    (n/scal! 0.0 L)
    (n/scal! 0.0 U)
    
    (loop [k 0
           S S
           L L
           U U]
      (if (>= k max-iter)
        {:sparse-S S :low-rank-L L :precision-theta theta :iterations k 
         :acyclicity (acyclicity-score S)}
        (let [;; 1. Update S with Acyclicity Gradient Step
              ;; S = soft-threshold(Theta + L - U - beta/rho * grad_h(S), lambda / rho)
              grad (acyclicity-gradient S)
              temp-S (n/copy theta)
              _ (n/axpy! 1.0 L temp-S)
              _ (n/axpy! -1.0 U temp-S)
              _ (n/axpy! (- (/ beta rho)) grad temp-S)
              new-S (soft-threshold temp-S (/ lambda rho))
              
              ;; 2. Update L
              temp-L (n/copy new-S)
              _ (n/axpy! -1.0 theta temp-L)
              _ (n/axpy! 1.0 U temp-L)
              new-L (sv-threshold temp-L (/ gamma rho))
              
              ;; 3. Update U
              new-U (n/copy U)
              _ (n/axpy! 1.0 theta new-U)
              _ (n/axpy! -1.0 new-S new-U)
              _ (n/axpy! 1.0 new-L new-U)
              
              err (r/nrm2 (n/axpy! -1.0 S (n/copy new-S)))]
          
          (if (< err tol)
            {:sparse-S new-S :low-rank-L new-L :precision-theta theta 
             :iterations k :error err :acyclicity (acyclicity-score new-S)}
            (recur (inc k) new-S new-L new-U)))))))

(defn learn-structure
  "Learns the causal structure from a precision matrix."
  [precision-matrix]
  (alvgl-decomposition precision-matrix 0.1 0.1))
