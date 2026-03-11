(ns consider.models-test
  (:require [clojure.test :refer :all]
            [consider.models :refer :all]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.vect-math :as vm]))

(deftest test-relu-activation
  (testing "relu! on vectors"
    (let [v (native/dv [-1.0 0.0 1.0 2.0])]
      (relu! v)
      (is (= [0.0 0.0 1.0 2.0] (vec (seq v))))))

  (testing "relu! on matrices"
    (let [m (native/dge 2 2 [(- 1.0) 0.0 ;; Column 0
                             2.0 (- 0.5)])] ;; Column 1
      (relu! m)
      (is (= [0.0 2.0] (mapv #(n/entry m 0 %) (range 2))) "Row 0")
      (is (= [0.0 0.0] (mapv #(n/entry m 1 %) (range 2))) "Row 1")))

  (testing "relu-grad on vectors"
    (let [v (native/dv [-1.0 0.5 1.0])
          g (relu-grad v)]
      (is (= [0.0 1.0 1.0] (vec (seq g))))))

  (testing "relu-grad on matrices"
    (let [m (native/dge 2 2 [(- 1.0) 1.0 ;; Column 0
                             0.5 0.0])] ;; Column 1
      (let [g (relu-grad m)]
        (is (= [0.0 1.0] (mapv #(n/entry g 0 %) (range 2))) "Grad Row 0")
        (is (= 1.0 (n/entry g 1 0)) "Grad [1,0]")
        (is (= 0.0 (n/entry g 1 1)) "Grad [1,1]")))))

(deftest test-predict-velocity-clipping
  (testing "predict-velocity clips output to [-10, 10]"
    (let [net (make-mlp-vector-field 1 1 4)
          _ (n/scal! 100.0 (:w2 net))
          x (native/dv [1.0])
          obs (native/dv [1.0])
          out (predict-velocity net x 0.5 obs)]
      (is (<= (n/entry out 0) 10.0))
      (is (>= (n/entry out 0) -10.0)))))

(deftest test-train-batch-loss-reduction
  (testing "train-batch! reduces loss on simple linear dataset"
    (let [state-dim 1
          obs-dim 1
          hidden-dim 8
          net (make-mlp-vector-field state-dim obs-dim hidden-dim)

          batch-size 4
          inputs (native/dge batch-size 3 [1.0 0.5 1.0
                                           2.0 0.5 2.0
                                           3.0 0.5 3.0
                                           4.0 0.5 4.0] {:layout :row})
          targets (native/dge batch-size 1 [2.0 4.0 6.0 8.0] {:layout :row})

          calc-loss (fn [curr-net]
                      (let [h1-pre (n/mm inputs (n/trans (:w1 curr-net)))
                            _ (dotimes [i batch-size] (n/axpy! 1.0 (:b1 curr-net) (n/row h1-pre i)))
                            h1 (relu! (n/copy h1-pre))
                            out (n/mm h1 (n/trans (:w2 curr-net)))
                            _ (dotimes [i batch-size] (n/axpy! 1.0 (:b2 curr-net) (n/row out i)))
                            diff (n/copy out)]
                        (n/axpy! -1.0 targets diff)
                        (n/dot diff diff)))]

      (let [initial-loss (calc-loss net)]
        (train-batch! net inputs targets 0.01 100)
        (let [final-loss (calc-loss net)]
          (is (< final-loss initial-loss) "Loss should decrease after training"))))))

(deftest test-grow-network-dimensions
  (testing "grow-network expands layers correctly while preserving weights"
    (let [old-state-dim 1
          old-obs-dim 1
          hidden-dim 4
          net (make-mlp-vector-field old-state-dim old-obs-dim hidden-dim)

          _ (n/entry! (:w1 net) 0 0 42.0) ;; First state weight
          _ (n/entry! (:w1 net) 0 1 7.0) ;; Time weight
          _ (n/entry! (:w1 net) 0 2 13.0) ;; First obs weight

          new-state-dim 2
          new-obs-dim 2
          expanded-net (grow-network net new-state-dim new-obs-dim)]

      (is (= hidden-dim (n/mrows (:w1 expanded-net))))
      (is (= 5 (n/ncols (:w1 expanded-net))))
      (is (= new-state-dim (n/mrows (:w2 expanded-net))))
      (is (= hidden-dim (n/ncols (:w2 expanded-net))))

      (is (= 42.0 (n/entry (:w1 expanded-net) 0 0)) "Old state weight preserved")
      (is (= 7.0 (n/entry (:w1 expanded-net) 0 2)) "Old time weight shifted")
      (is (= 13.0 (n/entry (:w1 expanded-net) 0 3)) "Old obs weight shifted"))))
