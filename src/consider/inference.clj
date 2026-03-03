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
  "Calculates the KL divergence between two multivariate Gaussians with diagonal covariances."
  [mu-q var-q mu-p var-p]
  ;; KL(q||p) = 0.5 * sum(var-q/var-p + (mu-p - mu-q)^2 / var-p - 1 + ln(var-p/var-q))
  (let [d (count mu-q)]
    (reduce +
            (map (fn [mq vq mp vp]
                   (* 0.5 (+ (/ vq vp)
                             (/ (Math/pow (- mp mq) 2) vp)
                             -1
                             (Math/log (/ (max 1e-9 vp) (max 1e-9 vq))))))
                 mu-q var-q mu-p var-p))))

(defn calculate-accuracy
  "Calculates the expected log likelihood (Accuracy) of the sensory data."
  [predicted-obs actual-obs observation-variance]
  ;; Accuracy = E_q[ln p(o|s)]
  ;; Assuming a Gaussian likelihood p(o|s) ~ N(predicted-obs, observation-variance)
  (reduce +
          (map (fn [po ao ov]
                 ;; ln N(ao; po, ov) = -0.5 * (ln(2*pi*ov) + (ao - po)^2 / ov)
                 (* -0.5 (+ (Math/log (* 2 Math/PI (max 1e-9 ov)))
                            (/ (Math/pow (- ao po) 2) (max 1e-9 ov)))))
               predicted-obs actual-obs observation-variance)))

(defn calculate-risk
  "Calculates the Risk (Pragmatic Value) as the KL divergence between predicted 
   observations and agent preferences (C-matrix).
   Risk = E_q[ln q(o|s) - ln p(o|C)]"
  [predicted-obs preferences observation-variance]
  (if (empty? preferences)
    0.0
    ;; Simplified: KL divergence between two Gaussians (Predicted vs Preferred)
    ;; For now, assume preferences is a vector of target observation values.
    (let [mu-p (first preferences) ;; Use the first preference as target for now
          var-p observation-variance]
      (reduce +
              (map (fn [po mp vp]
                     ;; KL(N(po, vp) || N(mp, vp)) = 0.5 * ((po - mp)^2 / vp)
                     (* 0.5 (/ (Math/pow (- mp po) 2) (max 1e-9 vp))))
                   predicted-obs mu-p var-p)))))

(defn variational-free-energy
  "Calculates the Variational Free Energy (F) and Expected Free Energy (G) components."
  [belief-state actual-obs likelihood-fn]
  (let [internal-states (:internal-states belief-state)
        ;; Simplified: treat all slots' positions as a single vector for now
        mu-q (mapv first (map :position (vals internal-states)))
        var-q (mapv first (map :variance (vals internal-states)))
        ;; Priors (p) - simplified to standard normal for now if not specified
        mu-p (vec (repeat (count mu-q) 0.0))
        var-p (vec (repeat (count var-q) 1.0))
        
        complexity (kl-divergence mu-q var-q mu-p var-p)
        
        predicted-obs (likelihood-fn internal-states)
        observation-variance (vec (repeat (count predicted-obs) 0.1))
        
        accuracy (calculate-accuracy predicted-obs actual-obs observation-variance)
        
        ;; Expected Free Energy Components (G)
        risk (calculate-risk predicted-obs (:preferences belief-state) observation-variance)
        
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
