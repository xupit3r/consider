(ns consider.inference
  "Implementation of Perceptual Inference (Variational Free Energy & Belief Updating)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.inference :as inf-spec]
            [consider.world-model :as wm]
            [consider.models :as models]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.random :as rng]
            [uncomplicate.neanderthal.real :as r]
            [uncomplicate.neanderthal.vect-math :as vm]))

;; --- ODE Solvers for Continuous Flows ---

(defn euler-step
  "Perform a single Euler step: x_{t+dt} = x_t + v(x_t, t) * dt."
  [v-fn x t dt]
  (let [velocity (v-fn x t)
        res (n/copy x)]
    (n/axpy! dt velocity res)
    res))

(defn integrate-ode
  "Integrate the ODE dx/dt = v(x, t) from t0 to t1 using Euler steps."
  [v-fn x0 t0 t1 steps]
  (let [dt (/ (- t1 t0) (double (max 1 steps)))]
    (loop [curr-x x0
           curr-t t0
           i 0]
      (if (>= i steps)
        curr-x
        (recur (euler-step v-fn curr-x curr-t dt)
               (+ curr-t dt)
               (inc i))))))

;; --- Flow Matching & Perceptual Inference ---

(defn flow-matching-sample
  "Generates a sample from q(s|o) by integrating a vector field from noise."
  [vector-field-fn context noise-sample steps]
  (let [v-fn (if (fn? vector-field-fn)
               (fn [x t] (vector-field-fn x t context))
               (fn [x t]
                 (let [obs-v (native/dv (count (:observation context)))]
                   (dotimes [i (n/dim obs-v)] (n/entry! obs-v i (double (nth (:observation context) i))))
                   (models/predict-velocity vector-field-fn x t obs-v))))]
    (integrate-ode v-fn noise-sample 0.0 1.0 steps)))

(defn kl-divergence
  "Calculates the KL divergence between two multivariate Gaussians with diagonal covariances using Neanderthal."
  [mu-q var-q mu-p var-p]
  (let [d (n/dim mu-q)
        ones (native/dv d)
        _ (dotimes [i d] (n/entry! ones i 1.0))

        t1 (n/copy var-q)
        _ (vm/div! t1 var-p)

        t2 (n/copy mu-p)
        _ (n/axpy! -1.0 mu-q t2)
        _ (vm/sqr! t2)
        _ (vm/div! t2 var-p)

        t3 (n/copy var-p)
        _ (vm/div! t3 var-q)
        _ (vm/log! t3)

        res (n/copy t1)
        _ (n/axpy! 1.0 t2 res)
        _ (n/axpy! 1.0 t3 res)
        _ (n/axpy! -1.0 ones res)]
    (* 0.5 (n/dot res ones))))

(defn calculate-accuracy
  "Calculates the expected log likelihood (Accuracy) of the sensory data using Neanderthal.
   Clips extreme values to prevent numerical instability."
  [predicted-obs actual-obs observation-variance]
  (let [d (n/dim predicted-obs)
        ones (native/dv d)
        _ (dotimes [i d] (n/entry! ones i 1.0))

        t1 (n/copy observation-variance)
        _ (n/scal! (* 2.0 Math/PI) t1)
        _ (vm/log! t1)

        t2 (n/copy actual-obs)
        _ (n/axpy! -1.0 predicted-obs t2)
        _ (vm/sqr! t2)
        _ (vm/div! t2 observation-variance)

        res (n/copy t1)
        _ (n/axpy! 1.0 t2 res)]
    ;; Clip individual log-likelihoods before summing
    (dotimes [i d]
      (n/entry! res i (max -1000.0 (min 1000.0 (n/entry res i)))))
    (* -0.5 (n/dot res ones))))

(defn calculate-risk
  "Calculates the Risk (Pragmatic Value) as the KL divergence between predicted 
   observations and agent preferences (C-matrix) using Neanderthal.
   Clips extreme values for stability."
  [predicted-obs preferences observation-variance]
  (if (empty? preferences)
    0.0
    (let [d (n/dim predicted-obs)
          mu-p (native/dv d)
          pref-data (first preferences)
          pref-count (count pref-data)
          _ (dotimes [i d]
              (n/entry! mu-p i (if (< i pref-count)
                                 (double (nth pref-data i))
                                 0.0)))

          diff (n/copy predicted-obs)
          _ (n/axpy! -1.0 mu-p diff)
          _ (vm/sqr! diff)
          _ (vm/div! diff observation-variance)
          ones (native/dv d)
          _ (dotimes [i d] (n/entry! ones i 1.0))]
      (max 0.0 (min 1000.0 (* 0.5 (n/dot diff ones)))))))

(defn variational-free-energy
  "Calculates the Variational Free Energy (F) and Expected Free Energy (G) components."
  [belief-state actual-obs likelihood-fn]
  (let [internal-states (:internal-states belief-state)
        sorted-slots (sort-by key internal-states)
        mu-q-data (mapcat #(:position (second %)) sorted-slots)
        var-q-data (mapcat #(:variance (second %)) sorted-slots)

        d (count mu-q-data)
        mu-q (native/dv d)
        var-q (native/dv d)
        _ (dotimes [i d]
            (n/entry! mu-q i (double (nth mu-q-data i)))
            (n/entry! var-q i (double (nth var-q-data i))))

        mu-p (native/dv d)
        var-p (native/dv d)
        _ (n/scal! 0.0 mu-p)
        _ (n/scal! 0.0 var-p)
        _ (dotimes [i d] (n/entry! var-p i 10.0))

        complexity (kl-divergence mu-q var-q mu-p var-p)

        predicted-obs-data (likelihood-fn internal-states)
        obs-dim (count predicted-obs-data)
        predicted-obs (native/dv obs-dim)
        actual-obs-v (native/dv obs-dim)
        obs-var (native/dv obs-dim)
        actual-obs-count (count actual-obs)
        _ (dotimes [i obs-dim]
            (n/entry! predicted-obs i (double (nth predicted-obs-data i)))
            (n/entry! actual-obs-v i (if (< i actual-obs-count)
                                       (double (nth actual-obs i))
                                       (double (nth predicted-obs-data i))))
            (n/entry! obs-var i 0.1))

        accuracy (calculate-accuracy predicted-obs actual-obs-v obs-var)
        risk (calculate-risk predicted-obs (:preferences belief-state) obs-var)

        elbo (- accuracy complexity)]
    {:elbo elbo
     :complexity complexity
     :accuracy accuracy
     :vfe (- complexity accuracy)
     :risk risk}))

(defn calculate-ambiguity
  "Calculates the Ambiguity (Epistemic Value) as the expected entropy of the observation model."
  [observation-variance]
  (let [d (n/dim observation-variance)
        ones (native/dv d)
        _ (dotimes [i d] (n/entry! ones i 1.0))

        t1 (n/copy observation-variance)
        _ (n/scal! (* 2.0 Math/PI Math/E) t1)
        _ (vm/log! t1)]
    (* 0.5 (n/dot t1 ones))))

(defn expected-free-energy
  "Calculates the Expected Free Energy (G) for a predicted belief state."
  [belief-state likelihood-fn]
  (let [internal-states (:internal-states belief-state)
        predicted-obs-data (likelihood-fn internal-states)
        obs-dim (count predicted-obs-data)
        predicted-obs (native/dv obs-dim)
        obs-var (native/dv obs-dim)
        _ (dotimes [i obs-dim]
            (n/entry! predicted-obs i (double (nth predicted-obs-data i)))
            (n/entry! obs-var i 0.1))

        risk (calculate-risk predicted-obs (:preferences belief-state) obs-var)
        ambiguity (calculate-ambiguity obs-var)]
    {:g (+ risk ambiguity)
     :risk risk
     :ambiguity ambiguity}))

(defn belief-update
  "Performs belief updating using Flow Matching (amortized inference) on the full state space."
  [belief-state actual-obs likelihood-fn vector-field-fn steps]
  (let [context {:observation actual-obs}
        internal-states (:internal-states belief-state)
        sorted-ids (sort (keys internal-states))

        prev-pos-data (mapcat #(:position (get internal-states %)) sorted-ids)
        total-dim (count prev-pos-data)

        noise (native/dv total-dim)
        _ (rng/rand-normal! noise)
        _ (dotimes [i total-dim]
            (n/entry! noise i (+ (n/entry noise i) (double (nth prev-pos-data i)))))

        updated-state-vec (flow-matching-sample vector-field-fn context noise steps)

        ;; Ensure we handle dimension mismatch gracefully
        actual-total-dim (n/dim updated-state-vec)

        updated-internal-states
        (loop [ids sorted-ids
               offset 0
               acc {}]
          (if (and (seq ids) (< offset actual-total-dim))
            (let [id (first ids)
                  old-slot (get internal-states id)
                  dim (count (:position old-slot))
                  new-pos (mapv #(n/entry updated-state-vec (+ offset %)) (range (min dim (- actual-total-dim offset))))]
              (recur (rest ids)
                     (+ offset dim)
                     (assoc acc id (assoc old-slot :position new-pos))))
            acc))

        limit (or (:history-limit belief-state) 100)
        updated-bs-temp (assoc belief-state :internal-states updated-internal-states)
        metrics (variational-free-energy updated-bs-temp actual-obs likelihood-fn)

        history-entry {:internal-states updated-internal-states
                       :observation actual-obs
                       :vfe (:vfe metrics)}
        history (conj (or (:history belief-state) []) history-entry)
        trimmed-history (if (> (count history) limit)
                          (subvec history (- (count history) limit))
                          history)

        updated-bs (assoc updated-bs-temp :history trimmed-history)]
    (merge updated-bs
           {:variational-free-energy (:vfe metrics)
            :efe-components {:risk (:risk metrics)
                             :ambiguity 0.0}})))

(defn- sample-prioritized-history
  "Samples a history entry with probability proportional to its VFE (surprise)."
  [history]
  (let [vfes (map #(or (:vfe %) 1.0) history)
        min-vfe (apply min vfes)
        probs (map #(+ 1e-6 (- % min-vfe)) vfes)
        total (reduce + probs)
        target (rand total)]
    (loop [acc 0.0
           items (map vector history probs)]
      (let [[item p] (first items)
            new-acc (+ acc p)]
        (if (or (>= new-acc target) (empty? (rest items)))
          item
          (recur new-acc (rest items)))))))

(defn- generate-training-sample
  "Generates a single Flow Matching training sample [input-v, target-v]."
  [x1 obs state-dim obs-dim]
  (let [x0 (native/dv state-dim)
        _ (rng/rand-normal! x0)
        t (rand)
        xt (n/copy x0)
        _ (n/scal! (- 1.0 t) xt)
        _ (n/axpy! t x1 xt)

        v-target (n/copy x1)
        _ (n/axpy! -1.0 x0 v-target)

        input-v (native/dv (+ state-dim 1 obs-dim))]
    (dotimes [i state-dim] (n/entry! input-v i (n/entry xt i)))
    (n/entry! input-v state-dim (double t))
    (dotimes [i obs-dim] (n/entry! input-v (+ state-dim 1 i) (n/entry obs i)))
    [input-v v-target]))

(defn train-recognition-model
  "Sleep Phase: Amortizes inference using Prioritized Experience Replay and Generative Replay ('Dreaming')."
  [vector-field-fn belief-state iterations]
  (if (fn? vector-field-fn)
    (do (println "Warning: vector-field-fn is a function, not a trainable network. Skipping amortization.")
        vector-field-fn)
    (let [history (:history belief-state)
          n-history (count history)
          batch-size 16]
      (let [state-dim (n/mrows (:w2 vector-field-fn))
            obs-dim (- (n/ncols (:w1 vector-field-fn)) state-dim 1)]
        (dotimes [_ iterations]
          (let [samples (repeatedly
                         batch-size
                         (fn []
                           (if (and (> n-history 0) (< (rand) 0.5))
                             ;; 1. Real History (Prioritized)
                             (let [item (sample-prioritized-history history)
                                   sorted-slots (sort-by key (:internal-states item))
                                   x1-data (mapcat #(:position (second %)) sorted-slots)
                                   x1 (native/dv state-dim)
                                   _ (dotimes [i (min state-dim (count x1-data))]
                                       (n/entry! x1 i (double (nth x1-data i))))
                                   obs-v (native/dv obs-dim)
                                   actual-obs (:observation item)
                                   _ (dotimes [i (min obs-dim (count actual-obs))]
                                       (n/entry! obs-v i (double (nth actual-obs i))))]
                               (generate-training-sample x1 obs-v state-dim obs-dim))
                             ;; 2. Generative Replay (Dreaming)
                             (let [dream (wm/dream-trajectory belief-state 1)
                                   item (first dream)
                                   sorted-slots (sort-by key (:internal-states item))
                                   x1-data (mapcat #(:position (second %)) sorted-slots)
                                   x1 (native/dv state-dim)
                                   _ (dotimes [i (min state-dim (count x1-data))]
                                       (n/entry! x1 i (double (nth x1-data i))))
                                   obs-data (:observation item)
                                   obs-v (native/dv obs-dim)
                                   _ (dotimes [i (min obs-dim (count obs-data))]
                                       (n/entry! obs-v i (double (nth obs-data i))))]
                               (generate-training-sample x1 obs-v state-dim obs-dim)))))

                input-mat (native/dge batch-size (+ state-dim 1 obs-dim))
                target-mat (native/dge batch-size state-dim)]

            (dotimes [i batch-size]
              (let [[in-v out-v] (nth samples i)]
                (n/copy! in-v (n/row input-mat i))
                (n/copy! out-v (n/row target-mat i))))

            (models/train-batch! vector-field-fn input-mat target-mat 0.01 1)))
        vector-field-fn))))
