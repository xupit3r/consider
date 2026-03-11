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
  ;; KL(q||p) = 0.5 * sum(var-q/var-p + (mu-p - mu-q)^2 / var-p - 1 + ln(var-p/var-q))
  (let [d (n/dim mu-q)
        ones (native/dv d)
        _ (dotimes [i d] (n/entry! ones i 1.0)) ;; Initialize to ones

        ;; term1: var-q / var-p
        t1 (n/copy var-q)
        _ (vm/div! t1 var-p)

        ;; term2: (mu-p - mu-q)^2 / var-p
        t2 (n/copy mu-p)
        _ (n/axpy! -1.0 mu-q t2)
        _ (vm/sqr! t2)
        _ (vm/div! t2 var-p)

        ;; term3: ln(var-p / var-q)
        t3 (n/copy var-p)
        _ (vm/div! t3 var-q)
        _ (vm/log! t3)

        ;; Sum all terms
        res (n/copy t1)
        _ (n/axpy! 1.0 t2 res)
        _ (n/axpy! 1.0 t3 res)
        _ (n/axpy! -1.0 ones res)]
    (* 0.5 (n/dot res ones))))

(defn calculate-accuracy
  "Calculates the expected log likelihood (Accuracy) of the sensory data using Neanderthal."
  [predicted-obs actual-obs observation-variance]
  ;; Accuracy = E_q[ln p(o|s)]
  ;; ln N(ao; po, ov) = -0.5 * (ln(2*pi*ov) + (ao - po)^2 / ov)
  (let [d (n/dim predicted-obs)
        ones (native/dv d)
        _ (dotimes [i d] (n/entry! ones i 1.0))

        ;; term1: ln(2*pi*ov)
        t1 (n/copy observation-variance)
        _ (n/scal! (* 2.0 Math/PI) t1)
        _ (vm/log! t1)

        ;; term2: (ao - po)^2 / ov
        t2 (n/copy actual-obs)
        _ (n/axpy! -1.0 predicted-obs t2)
        _ (vm/sqr! t2)
        _ (vm/div! t2 observation-variance)

        ;; Sum terms
        res (n/copy t1)
        _ (n/axpy! 1.0 t2 res)]
    (* -0.5 (n/dot res ones))))

(defn calculate-risk
  "Calculates the Risk (Pragmatic Value) as the KL divergence between predicted 
   observations and agent preferences (C-matrix) using Neanderthal.
   Robustly handles cases where predicted observations have more dimensions than preferences."
  [predicted-obs preferences observation-variance]
  (if (empty? preferences)
    0.0
    (let [d (n/dim predicted-obs)
          mu-p (native/dv d)
          ;; For now, use the first preference as target
          pref-data (first preferences)
          pref-count (count pref-data)
          _ (dotimes [i d]
              (n/entry! mu-p i (if (< i pref-count)
                                 (double (nth pref-data i))
                                 0.0))) ;; Pad with zeros for novel slots

          ;; Risk = 0.5 * ((po - mp)^2 / vp)
          diff (n/copy predicted-obs)
          _ (n/axpy! -1.0 mu-p diff)
          _ (vm/sqr! diff)
          _ (vm/div! diff observation-variance)
          ones (native/dv d)
          _ (dotimes [i d] (n/entry! ones i 1.0))]
      (* 0.5 (n/dot diff ones)))))

(defn variational-free-energy
  "Calculates the Variational Free Energy (F) and Expected Free Energy (G) components."
  [belief-state actual-obs likelihood-fn]
  (let [internal-states (:internal-states belief-state)
        ;; Aggregate slots into Neanderthal vectors
        sorted-slots (sort-by key internal-states)
        mu-q-data (mapv #(first (:position (second %))) sorted-slots)
        var-q-data (mapv #(first (:variance (second %))) sorted-slots)

        d (count mu-q-data)
        mu-q (native/dv d)
        var-q (native/dv d)
        _ (dotimes [i d]
            (n/entry! mu-q i (double (nth mu-q-data i)))
            (n/entry! var-q i (double (nth var-q-data i))))

        ;; Standard normal prior: mu=0, var=1
        mu-p (native/dv d)
        var-p (native/dv d)
        _ (n/scal! 0.0 mu-p)
        _ (n/scal! 0.0 var-p)
        _ (dotimes [i d] (n/entry! var-p i 1.0))

        complexity (kl-divergence mu-q var-q mu-p var-p)

        predicted-obs-data (likelihood-fn internal-states)
        obs-dim (count predicted-obs-data)
        predicted-obs (native/dv obs-dim)
        actual-obs-v (native/dv obs-dim)
        obs-var (native/dv obs-dim)
        _ (dotimes [i obs-dim]
            (n/entry! predicted-obs i (double (nth predicted-obs-data i)))
            (n/entry! actual-obs-v i (double (nth actual-obs i)))
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
  "Calculates the Ambiguity (Epistemic Value) as the expected entropy of 
   the observation model: H[p(o|s)] = E_q[ln p(o|s)].
   For a Gaussian observation model, this is proportional to the log variance."
  [observation-variance]
  (let [d (n/dim observation-variance)
        ones (native/dv d)
        _ (dotimes [i d] (n/entry! ones i 1.0))

        ;; H = 0.5 * sum(ln(2 * pi * e * ov))
        t1 (n/copy observation-variance)
        _ (n/scal! (* 2.0 Math/PI Math/E) t1)
        _ (vm/log! t1)]
    (* 0.5 (n/dot t1 ones))))

(defn expected-free-energy
  "Calculates the Expected Free Energy (G) for a predicted belief state.
   G = Risk + Ambiguity."
  [belief-state likelihood-fn]
  (let [internal-states (:internal-states belief-state)
        predicted-obs-data (likelihood-fn internal-states)
        obs-dim (count predicted-obs-data)
        predicted-obs (native/dv obs-dim)
        obs-var (native/dv obs-dim)
        _ (dotimes [i obs-dim]
            (n/entry! predicted-obs i (double (nth predicted-obs-data i)))
            (n/entry! obs-var i 0.1)) ;; Default observation variance

        risk (calculate-risk predicted-obs (:preferences belief-state) obs-var)
        ambiguity (calculate-ambiguity obs-var)]
    {:g (+ risk ambiguity)
     :risk risk
     :ambiguity ambiguity}))

(defn belief-update
  "Performs belief updating using Flow Matching (amortized inference).
   Also maintains a history of internal states and observations for causal structure discovery and model training."
  [belief-state actual-obs likelihood-fn vector-field-fn steps]
  (let [context {:observation actual-obs}
        updated-internal-states
        (reduce-kv
         (fn [m id slot]
           (let [dim (count (:position slot))
                 noise (native/dv dim)]
             (rng/rand-normal! noise)
             (let [pos-sample (flow-matching-sample vector-field-fn context noise steps)
                   dim (n/dim pos-sample)
                   new-pos (mapv (fn [i] (n/entry pos-sample i)) (range dim))]
               (assoc m id (assoc slot :position new-pos)))))
         {}
         (:internal-states belief-state))

        limit (or (:history-limit belief-state) 100)
        ;; Store both internal states and the observation that led to them
        history-entry {:internal-states updated-internal-states
                       :observation actual-obs}
        history (conj (or (:history belief-state) []) history-entry)
        trimmed-history (if (> (count history) limit)
                          (subvec history (- (count history) limit))
                          history)

        updated-bs (assoc belief-state
                          :internal-states updated-internal-states
                          :history trimmed-history)
        metrics (variational-free-energy updated-bs actual-obs likelihood-fn)]
    (merge updated-bs
           {:variational-free-energy (:vfe metrics)
            :efe-components {:risk (:risk metrics)
                             :ambiguity 0.0}})))

(defn train-recognition-model
  "Sleep Phase: Amortizes inference by training the vector-field-fn (Recognition Model)
   to match the flows generated by the World Model using Flow Matching.
   Randomly samples from belief history to ensure robust amortization."
  [vector-field-fn belief-state iterations]
  (if (fn? vector-field-fn)
    ;; Fallback for simulation if not a neural network
    (do
      (println "Warning: vector-field-fn is a function, not a trainable network. Skipping amortization.")
      vector-field-fn)
    (let [history (:history belief-state)
          n-history (count history)]
      (if (zero? n-history)
        vector-field-fn
        (do
          (dotimes [_ iterations]
            (let [;; 1. Randomly sample a historical experience
                  sample (rand-nth history)
                  internal-states (:internal-states sample)
                  actual-obs (:observation sample)

                  ;; 2. Aggregate internal states into a single target vector x1
                  sorted-slots (sort-by key internal-states)
                  x1-data (mapv #(first (:position (second %))) sorted-slots)
                  d (count x1-data)

                  ;; Check if dimensions match the network (handle Growth)
                  ;; Note: In a production system, we'd use a dynamic or slot-wise network.
                  network-state-dim (n/mrows (:w2 vector-field-fn))]

              (when (= d network-state-dim)
                (let [x1 (native/dv d)
                      _ (dotimes [i d] (n/entry! x1 i (double (nth x1-data i))))

                      ;; 3. Flow Matching: Sample x0 and interpolate
                      x0 (native/dv d)
                      _ (rng/rand-normal! x0)
                      t (rand)
                      xt (n/copy x0)
                      _ (n/scal! (- 1.0 t) xt)
                      _ (n/axpy! t x1 xt)

                      ;; Target velocity: v = x1 - x0
                      v-target (n/copy x1)
                      _ (n/axpy! -1.0 x0 v-target)

                      ;; Observation context
                      obs-dim (count actual-obs)
                      network-obs-dim (- (n/ncols (:w1 vector-field-fn)) d 1)]

                  (when (= obs-dim network-obs-dim)
                    (let [obs-v (native/dv obs-dim)
                          _ (dotimes [i obs-dim] (n/entry! obs-v i (double (nth actual-obs i))))

                          ;; Construct network input: [xt, t, obs-v]
                          input-v (native/dv (+ d 1 obs-dim))
                          _ (dotimes [i d] (n/entry! input-v i (n/entry xt i)))
                          _ (n/entry! input-v d (double t))
                          _ (dotimes [i obs-dim] (n/entry! input-v (+ d 1 i) (n/entry obs-v i)))]

                      ;; 4. Update parameters using manual SGD in models.clj
                      (models/train-on-samples! vector-field-fn input-v v-target 0.01 1)))))))
          vector-field-fn)))))
