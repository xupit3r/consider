(ns consider.inference
  "Implementation of Perceptual Inference (Variational Free Energy & Belief Updating)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.inference :as inf-spec]
            [consider.world-model :as wm]
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
  (let [v-fn (fn [x t] (vector-field-fn x t context))]
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
   observations and agent preferences (C-matrix) using Neanderthal."
  [predicted-obs preferences observation-variance]
  (if (empty? preferences)
    0.0
    (let [mu-p (native/dv (n/dim predicted-obs))
          ;; For now, use the first preference as target
          pref-data (first preferences)
          _ (dotimes [i (n/dim mu-p)] (n/entry! mu-p i (double (nth pref-data i))))
          
          ;; Risk = 0.5 * ((po - mp)^2 / vp)
          diff (n/copy predicted-obs)
          _ (n/axpy! -1.0 mu-p diff)
          _ (vm/sqr! diff)
          _ (vm/div! diff observation-variance)          
          ones (native/dv (n/dim predicted-obs))
          _ (dotimes [i (n/dim ones)] (n/entry! ones i 1.0))]
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
        
        mu-p (native/dv d) ;; Standard normal prior
        var-p (native/dv d)
        _ (n/scal! 0.0 mu-p)
        _ (n/scal! 0.0 var-p)
        _ (n/axpy! 1.0 (native/dv d) var-p)
        
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

(defn belief-update
  "Performs belief updating using Flow Matching (amortized inference)."
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
        
        updated-bs (assoc belief-state :internal-states updated-internal-states)
        metrics (variational-free-energy updated-bs actual-obs likelihood-fn)]
    (merge updated-bs 
           {:variational-free-energy (:vfe metrics)
            :efe-components {:risk (:risk metrics)
                             :ambiguity 0.0}})))

(defn train-recognition-model
  "Sleep Phase: Amortizes inference by training the vector-field-fn (Recognition Model)
   to match the flows generated by the World Model.
   In this implementation, we simulate a parameter update using simple gradient 
   descent on the velocity field error."
  [vector-field-fn belief-state iterations]
  (let [internal-states (:internal-states belief-state)]
    (dotimes [i iterations]
      ;; Simulate training: Find the current velocity field's error relative 
      ;; to a target 'generative' velocity and update 'parameters'.
      (let [target-velocity 0.1 ;; Simplified target
            current-velocity (first (mapv first (map :position (vals internal-states))))
            error (- target-velocity current-velocity)
            learning-rate 0.01]
        ;; In a real neural model, we would use (opt/step! optimizer loss-grad)
        (if (> (Math/abs error) 1e-3)
          (do 
            ;; Simulated parameter update log
            (when (zero? (mod i 5))
              (println "Training iteration" i "- Error:" (format "%.4f" (double error))))))))
    vector-field-fn))
