(ns consider.causal
  "Implementation of Causal Discovery (ALVGL & NOTEARS)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.causal :as causal-spec]
            [consider.specs.hierarchy :as hier-spec]
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
  "Calculates the NOTEARS acyclicity score.
   Uses a higher-order Taylor approximation for exp(W * W)."
  [W]
  (let [d (n/mrows W)
        W-sq (vm/sqr (n/copy W))
        I (native/dge d d)]
    (n/scal! 0.0 I)
    (dotimes [i d] (n/entry! I i i 1.0))
    (let [M1 W-sq
          M2 (n/mm M1 M1)
          M3 (n/mm M2 M1)
          M4 (n/mm M2 M2)
          exp-M (n/copy I)]
      (n/axpy! 1.0 M1 exp-M)
      (n/axpy! 0.5 M2 exp-M)
      (n/axpy! (/ 1.0 6.0) M3 exp-M)
      (n/axpy! (/ 1.0 24.0) M4 exp-M)
      (let [score (- (matrix-trace exp-M) d)]
        (if (or (Double/isNaN score) (Double/isInfinite score)) 100.0 score)))))

(defn acyclicity-gradient
  "Calculates the gradient of the NOTEARS acyclicity score."
  [W]
  (let [d (n/mrows W)
        W-sq (vm/sqr (n/copy W))
        I (native/dge d d)]
    (n/scal! 0.0 I)
    (dotimes [i d] (n/entry! I i i 1.0))
    (let [M1 W-sq
          M2 (n/mm M1 M1)
          M3 (n/mm M2 M1)
          exp-M-deriv (n/copy I)]
      (n/axpy! 1.0 M1 exp-M-deriv)
      (n/axpy! 0.5 M2 exp-M-deriv)
      (n/axpy! (/ 1.0 6.0) M3 exp-M-deriv)
      (let [grad (n/trans exp-M-deriv)
            two-W (n/copy W)]
        (n/scal! 2.0 two-W)
        (vm/mul! grad two-W)
        (let [gnorm (r/nrm2 grad)]
          (if (or (Double/isNaN gnorm) (> gnorm 100.0))
            (n/scal! (/ 10.0 (or gnorm 1.0)) grad)
            grad))))))

(defn alvgl-decomposition
  "Performs Sparse-Low Rank decomposition of a precision matrix using ADMM with DAG constraints."
  [theta lambda gamma & {:keys [rho beta max-iter tol eta]
                         :or {rho 5.0 beta 0.5 max-iter 200 tol 1e-4 eta 0.05}}]
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
          {:sparse-S S :low-rank-L L :precision-matrix theta :iterations k :acyclicity score})
        (let [grad (acyclicity-gradient S)
              temp-S (n/copy theta)
              _ (n/axpy! -1.0 L temp-S)
              _ (n/axpy! -1.0 U temp-S)
              _ (n/axpy! (- (* beta eta)) grad temp-S)
              new-S (soft-threshold temp-S (/ lambda rho))
              temp-L (n/copy theta)
              _ (n/axpy! -1.0 new-S temp-L)
              _ (n/axpy! -1.0 U temp-L)
              new-L (sv-threshold temp-L (/ gamma rho))
              new-U (n/copy U)
              _ (n/axpy! 1.0 new-S new-U)
              _ (n/axpy! 1.0 new-L new-U)
              _ (n/axpy! -1.0 theta new-U)
              diff (n/axpy! -1.0 S (n/copy new-S))
              err (r/nrm2 diff)]
          (if (or (< err tol) (Double/isNaN err))
            (let [score (acyclicity-score new-S)]
              {:sparse-S new-S :low-rank-L new-L :precision-matrix theta :iterations k :error err :acyclicity score})
            (recur (inc k) new-S new-L new-U)))))))

(defn cluster-causal-modules
  "Identifies strongly-connected modules in the causal adjacency matrix S.
   Uses a simple BFS to find connected components in the symmetrized adjacency graph."
  [S slot-ids threshold]
  (let [ids (vec (sort slot-ids))
        d (n/mrows S)
        n-ids (count ids)]
    (if (or (<= d 1) (<= n-ids 1))
      (if (seq ids) [(set ids)] [])
      (let [adj (n/copy S)
            _ (vm/abs! adj)
            sym-adj (n/axpy 1.0 (n/trans adj) adj)
            max-val (r/nrm2 sym-adj)
            t (if (> max-val 1.0) (* threshold max-val) threshold)]
        (loop [idx 0
               clusters []
               all-visited-indices #{}]
          (if (or (>= idx d) (>= idx n-ids))
            clusters
            (if (contains? all-visited-indices idx)
              (recur (inc idx) clusters all-visited-indices)
              (let [;; BFS
                    component (loop [queue [idx]
                                     visited #{idx}]
                                (if (empty? queue)
                                  visited
                                  (let [curr (first queue)
                                        neighbors (reduce (fn [acc j]
                                                            (if (and (< j d) (< j n-ids)
                                                                     (not (contains? visited j))
                                                                     (> (n/entry sym-adj curr j) t))
                                                              (conj acc j)
                                                              acc))
                                                          []
                                                          (range (min d n-ids)))]
                                    (recur (concat (rest queue) neighbors)
                                           (into visited neighbors)))))]
                (let [module-ids (set (map #(nth ids %) component))]
                  (recur (inc idx)
                         (conj clusters module-ids)
                         (into all-visited-indices component)))))))))))

(defn learn-hierarchy
  "Level 2 Learning: Groups slots into concepts based on causal dependencies.
   Clusters based on the learned low-rank precision matrix L."
  [causal-res slot-ids]
  (let [L (:low-rank-L causal-res)
        modules (cluster-causal-modules L slot-ids 0.1)]
    (map-indexed (fn [idx constituent-ids]
                   {:concept-id (keyword (str "concept-" idx))
                    :constituent-slots constituent-ids
                    :position [0.0]
                    :variance [1.0]})
                 modules)))

(defn learn-structure
  "Learns the causal structure from a precision matrix.
   Normalizes the precision matrix to ensure stable ADMM convergence."
  [precision-matrix]
  (let [d (n/mrows precision-matrix)
        norm-val (r/nrm2 precision-matrix)]
    (if (or (zero? norm-val) (Double/isNaN norm-val))
      (alvgl-decomposition precision-matrix 0.01 0.1)
      (let [normalized-theta (n/copy precision-matrix)]
        (n/scal! (/ 1.0 norm-val) normalized-theta)
        (alvgl-decomposition normalized-theta 0.01 0.1 :rho 5.0 :beta 0.5 :max-iter 200)))))
