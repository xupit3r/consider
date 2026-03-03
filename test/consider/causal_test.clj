(ns consider.causal-test
  (:require [clojure.test :refer :all]
            [consider.causal :refer :all]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]
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
  (let [theta (native/dge 2 2)]
    (n/scal! 1.0 theta)
    (let [decomp (alvgl-decomposition theta 0.1 0.1 :beta 0.1)]
      (is (contains? decomp :sparse-S))
      (is (contains? decomp :low-rank-L))
      (is (contains? decomp :acyclicity)))))
