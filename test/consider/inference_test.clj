(ns consider.inference-test
  (:require [clojure.test :refer :all]
            [consider.inference :refer :all]
            [consider.world-model :as wm]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [clojure.spec.alpha :as s]))

(deftest test-kl-divergence
  (let [mu-q [0.0] var-q [1.0]
        mu-p [0.0] var-p [1.0]]
    (is (zero? (kl-divergence mu-q var-q mu-p var-p)))))

(deftest test-calculate-accuracy
  (let [po [1.0] ao [1.0] ov [0.1]]
    ;; Accuracy is ln N(1.0; 1.0, 0.1) = -0.5 * (ln(2*pi*0.1) + 0)
    ;; 2 * pi * 0.1 ~ 0.628, ln(0.628) ~ -0.465, -0.5 * -0.465 ~ 0.232
    (is (> (calculate-accuracy po ao ov) 0.0))))

(deftest test-variational-free-energy
  (let [bs (-> (wm/make-belief-state)
               (wm/update-slot :e1 [0.0] [1.0]))
        likelihood-fn (fn [states] [0.0])
        actual-obs [0.1]
        vfe-metrics (variational-free-energy bs actual-obs likelihood-fn)]
    (is (contains? vfe-metrics :elbo))
    (is (contains? vfe-metrics :vfe))))

(deftest test-belief-update
  (let [bs (-> (wm/make-belief-state)
               (wm/update-slot :e1 [0.0] [1.0]))
        likelihood-fn (fn [states] [0.0])
        ;; Simple vector field: constant velocity towards observations
        vector-field-fn (fn [x t context]
                          (let [obs (first (get-in context [:observation]))
                                current-x (n/entry x 0)
                                velocity (native/dv (n/dim x))]
                            (n/entry! velocity 0 (- obs current-x))
                            velocity))
        actual-obs [0.1]
        updated-bs (belief-update bs actual-obs likelihood-fn vector-field-fn 10)]
    (is (not= bs updated-bs))
    (is (contains? updated-bs :variational-free-energy))))
