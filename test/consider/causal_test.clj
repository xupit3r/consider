(ns consider.causal-test
  (:require [clojure.test :refer :all]
            [consider.causal :refer :all]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.linalg :as la]
            [uncomplicate.neanderthal.random :as rng]
            [uncomplicate.neanderthal.real :as r]
            [clojure.spec.alpha :as s]))

(deftest test-acyclicity-score
  (let [W (native/dge 2 2)]
    (n/scal! 0.0 W)
    ;; For a zero matrix, h(W) = Tr(I) - d = 2 - 2 = 0
    (is (< (Math/abs (acyclicity-score W)) 1e-6))

    (n/entry! W 0 1 1.0)
    ;; For a DAG with one edge, h(W) should still be 0 (no cycles)
    (is (< (Math/abs (acyclicity-score W)) 1e-6))

    (n/entry! W 1 0 1.0)
    ;; For a matrix with a cycle, h(W) should be > 0
    (is (> (acyclicity-score W) 0.0))))

(deftest test-acyclicity-gradient
  (let [W (native/dge 2 2)]
    (n/scal! 0.0 W)
    (let [grad (acyclicity-gradient W)]
      ;; For zero matrix, exp(0) = I, grad = I^T ∘ 0 = 0
      (is (< (n/nrm2 grad) 1e-6)))

    (n/entry! W 0 1 1.0)
    (n/entry! W 1 0 1.0)
    (let [grad (acyclicity-gradient W)]
      ;; For a cycle, gradient should be non-zero
      (is (> (n/nrm2 grad) 0.0)))))

(deftest test-alvgl-decomposition
  (testing "2x2 case"
    (let [theta (native/dge 2 2)]
      (n/scal! 1.0 theta)
      (let [decomp (alvgl-decomposition theta 0.1 0.1 :beta 0.1)]
        (is (contains? decomp :sparse-S))
        (is (contains? decomp :low-rank-L))
        (is (contains? decomp :acyclicity)))))

  (testing "1x1 case"
    (let [theta (native/dge 1 1)]
      (n/entry! theta 0 0 1.0)
      (let [decomp (alvgl-decomposition theta 0.1 0.1 :beta 0.1)]
        (is (contains? decomp :sparse-S))
        (is (contains? decomp :low-rank-L))))))

(deftest test-causal-recovery
  (testing "Recovering a ground-truth linear causal link A -> B"
    (let [n-samples 200
          d 2
          X (native/dge n-samples d)
          rng (java.util.Random. 42)
          ;; Generate data: B = 0.5*A + noise
          _ (dotimes [i n-samples]
              (let [a (.nextGaussian rng)
                    b (+ (* 0.5 a) (* 0.2 (.nextGaussian rng)))]
                (n/entry! X i 0 a)
                (n/entry! X i 1 b)))

          ;; 1. Center X
          _ (let [means (native/dv d)]
              (n/scal! 0.0 means)
              (dotimes [j d]
                (let [col (n/submatrix X 0 j n-samples 1)]
                  (n/entry! means j (/ (n/asum col) n-samples))))
              (dotimes [i n-samples]
                (dotimes [j d]
                  (n/entry! X i j (- (n/entry X i j) (n/entry means j))))))

          ;; 2. Compute Precision Matrix Theta
          C (n/mm (n/trans X) X)
          _ (n/scal! (/ 1.0 (dec n-samples)) C)
          _ (dotimes [i d] (n/entry! C i i (+ (n/entry C i i) 0.01)))

          theta (let [I (native/dge d d)]
                  (n/scal! 0.0 I)
                  (dotimes [i d] (n/entry! I i i 1.0))
                  (la/sv C I))

          ;; Normalize Theta to avoid exploding gradients
          max-val (r/nrm2 theta)
          _ (n/scal! (/ 1.0 max-val) theta)]

      ;; 3. Learn Structure
      ;; Reduced lambda to 0.01 to allow more edges
      (let [decomp (alvgl-decomposition theta 0.01 0.01 :beta 0.1 :rho 10.0 :max-iter 200)
            S (:sparse-S decomp)]

        (let [s01 (Math/abs (n/entry S 0 1))
              s10 (Math/abs (n/entry S 1 0))]
          (println "Normalized Precision Matrix Theta:" theta)
          (println "Learned Sparse S:" S)
          (println "Acyclicity Score:" (:acyclicity decomp))
          (is (> (+ s01 s10) 0.01) "At least one directional edge should be identified")
          (is (< (:acyclicity decomp) 0.05) "The resulting structure must be a DAG"))))))
