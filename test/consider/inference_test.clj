(ns consider.inference-test
  (:require [clojure.test :refer :all]
            [consider.inference :refer :all]
            [consider.world-model :as wm]
            [consider.models :as models]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [clojure.spec.alpha :as s]))

(deftest test-kl-divergence
  (let [mu-q (native/dv 1)
        var-q (native/dv 1)
        mu-p (native/dv 1)
        var-p (native/dv 1)]
    (n/entry! mu-q 0 0.0)
    (n/entry! var-q 0 1.0)
    (n/entry! mu-p 0 0.0)
    (n/entry! var-p 0 1.0)
    (is (zero? (kl-divergence mu-q var-q mu-p var-p)))))

(deftest test-calculate-accuracy
  (let [po (native/dv 1)
        ao (native/dv 1)
        ov (native/dv 1)]
    (n/entry! po 0 1.0)
    (n/entry! ao 0 1.0)
    (n/entry! ov 0 0.1)
    ;; Accuracy is ln N(1.0; 1.0, 0.1) = -0.5 * (ln(2*pi*0.1) + 0)
    ;; 2 * pi * 0.1 ~ 0.628, ln(0.628) ~ -0.465, -0.5 * -0.465 ~ 0.232
    (is (> (calculate-accuracy po ao ov) 0.0))))

(deftest test-calculate-risk
  (let [po (native/dv 1)
        preferences [[2.0]]
        ov (native/dv 1)]
    (n/entry! po 0 1.0)
    (n/entry! ov 0 0.1)
    ;; Risk is KL(N(1.0, 0.1) || N(2.0, 0.1)) = 0.5 * (1.0 - 2.0)^2 / 0.1 = 0.5 * 1.0 / 0.1 = 5.0
    (is (= 5.0 (calculate-risk po preferences ov)))))

(deftest test-variational-free-energy
  (let [bs (-> (wm/make-belief-state {} [[2.0]])
               (wm/update-slot :e1 [1.0] [1.0]))
        likelihood-fn (fn [states] [(first (:position (get states :e1)))])
        actual-obs [1.0]
        vfe-metrics (variational-free-energy bs actual-obs likelihood-fn)]
    (is (contains? vfe-metrics :elbo))
    (is (contains? vfe-metrics :vfe))
    (is (= 5.0 (:risk vfe-metrics)))))

(deftest test-belief-update
  (let [bs (-> (wm/make-belief-state {} [[2.0]])
               (wm/update-slot :e1 [0.0] [1.0]))
        likelihood-fn (fn [states] [(first (:position (get states :e1)))])
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
    (is (contains? updated-bs :variational-free-energy))
    (is (contains? (:efe-components updated-bs) :risk))))

(deftest test-neural-training
  (let [state-dim 1
        obs-dim 1
        hidden-dim 4
        net (models/make-mlp-vector-field state-dim obs-dim hidden-dim)
        
        bs (-> (wm/make-belief-state {} [[2.0]])
               (wm/update-slot :e1 [1.0] [1.0])
               (wm/with-generative-model (fn [states] [1.0]) (fn [s a] s)))
        
        trained-net (train-recognition-model net bs 5)]
    (is (not (fn? trained-net)))
    (is (some? trained-net))
    (is (instance? consider.models.NeanderthalMLP trained-net))))
