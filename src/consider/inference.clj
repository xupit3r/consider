(ns consider.inference
  "Implementation of Perceptual Inference (Variational Free Energy & Belief Updating)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.inference :as inf-spec]
            [consider.world-model :as wm]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]))

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
                             (Math/log (/ vp vq)))))
                 mu-q var-q mu-p var-p))))

(defn calculate-accuracy
  "Calculates the expected log likelihood (Accuracy) of the sensory data."
  [predicted-obs actual-obs observation-variance]
  ;; Accuracy = E_q[ln p(o|s)]
  ;; Assuming a Gaussian likelihood p(o|s) ~ N(predicted-obs, observation-variance)
  (reduce +
          (map (fn [po ao ov]
                 ;; ln N(ao; po, ov) = -0.5 * (ln(2*pi*ov) + (ao - po)^2 / ov)
                 (* -0.5 (+ (Math/log (* 2 Math/PI ov))
                            (/ (Math/pow (- ao po) 2) ov))))
               predicted-obs actual-obs observation-variance)))

(defn variational-free-energy
  "Calculates the Variational Free Energy (F)."
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
        ;; Assume actual-obs and predicted-obs are vectors of floats for now
        accuracy (calculate-accuracy predicted-obs actual-obs (vec (repeat (count actual-obs) 0.1)))
        
        elbo (- accuracy complexity)]
    {:elbo elbo
     :complexity complexity
     :accuracy accuracy
     :vfe (- complexity accuracy)}))

(defn belief-update
  "Performs a simple gradient descent step on Variational Free Energy."
  [belief-state actual-obs likelihood-fn learning-rate]
  ;; In a real implementation, we would use Flow Matching or DDVI.
  ;; Here we simulate a small step that reduces VFE.
  (let [metrics (variational-free-energy belief-state actual-obs likelihood-fn)
        ;; Dummy update: just slightly nudge positions towards observations
        updated-bs (reduce-kv 
                    (fn [bs id slot]
                      (let [pos (:position slot)
                            new-pos (mapv (fn [p] (+ p (* learning-rate 0.1))) pos)] ;; Mock gradient
                        (wm/update-slot bs id new-pos (:variance slot))))
                    belief-state
                    (:internal-states belief-state))]
    (assoc updated-bs :variational-free-energy (:vfe metrics))))
