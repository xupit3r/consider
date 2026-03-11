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

(deftest test-grow-and-merge
  (let [bs (-> (make-belief-state)
               (grow-slots [(make-slot :e1 [1.0]) (make-slot :e2 [2.0])]))]
    (is (= 2 (count (:internal-states bs))))
    (let [merged-bs (merge-slots bs :e1 [:e2])]
      (is (= 1 (count (:internal-states merged-bs))))
      (is (contains? (:internal-states merged-bs) :e1))
      (is (not (contains? (:internal-states merged-bs) :e2))))))

(deftest test-generative-model-prediction
  (let [likelihood-fn (fn [states] [{:obs 1.0}])
        transition-fn (fn [states action] (update-in states [:e1 :position 0] inc))
        bs (-> (make-belief-state)
               (update-slot :e1 [0.0] [1.0])
               (with-generative-model likelihood-fn transition-fn))]
    (testing "Observation prediction"
      (is (= [{:obs 1.0}] (predict-observation bs))))
    (testing "Next state prediction"
      (let [next-states (predict-next-state bs :some-action)]
        (is (= 1.0 (first (get-in next-states [:e1 :position]))))))))

(deftest test-validate-belief-state
  (let [bs (make-belief-state)]
    (is (validate-belief-state bs))
    (is (thrown? Exception (validate-belief-state (assoc bs :internal-states {:e1 {:invalid :data}}))))))

(deftest test-novelty-identification
  (let [bs (-> (make-belief-state)
               (update-slot :e1 [10.0] [1.0]))]
    (testing "Detecting an unmodeled observation (Novel Entity)"
      ;; Case: actual observation has 2 objects, prediction only has 1
      (let [actual-obs [10.0 50.0]
            predicted-obs [10.0]
            new-slots (identify-novel-entities bs actual-obs predicted-obs)]
        (is (= 1 (count new-slots)))
        (is (= [50.0] (:position (first new-slots))))))))
