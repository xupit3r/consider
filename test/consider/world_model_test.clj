(ns consider.world-model-test
  (:require [clojure.test :refer :all]
            [consider.world-model :refer :all]
            [clojure.spec.alpha :as s]
            [consider.specs.world-model :as wm-spec]))

(deftest test-make-slot
  (let [slot (make-slot :entity-1 [1.0 2.0])]
    (is (= :entity-1 (:entity-id slot)))
    (is (= [1.0 2.0] (:position slot)))
    (is (= [1.0 1.0] (:variance slot)))))

(deftest test-make-belief-state
  (let [bs (make-belief-state)]
    (is (empty? (:internal-states bs)))
    (is (zero? (:variational-free-energy bs)))
    (is (zero? (:expected-free-energy bs)))
    (is (zero? (get-in bs [:efe-components :risk])))
    (is (zero? (get-in bs [:efe-components :ambiguity])))))

(deftest test-update-slot
  (let [bs (-> (make-belief-state)
               (update-slot :e1 [0.0] [0.1]))]
    (is (= [0.0] (:position (get-slot bs :e1))))
    (is (= [0.1] (:variance (get-slot bs :e1))))))

(deftest test-validate-belief-state
  (let [bs (make-belief-state)]
    (is (validate-belief-state bs))
    (is (thrown? Exception (validate-belief-state (assoc bs :internal-states {:e1 {:invalid :data}}))))))
