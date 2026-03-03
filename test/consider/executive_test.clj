(ns consider.executive-test
  (:require [clojure.test :refer :all]
            [consider.executive :refer :all]
            [consider.llm :as llm]
            [clojure.spec.alpha :as s]
            [consider.specs.executive :as exec-spec]))

(deftest test-make-node
  (let [node (make-node {:node-id "1" :parent-id nil :state [] :action "A" :prior-prob 1.0})]
    (is (= "1" (:node-id node)))
    (is (zero? (:visits node)))
    (is (= 1.0 (:prior-prob node)))))

(deftest test-make-initial-orchestrator-state
  (let [state (make-initial-orchestrator-state [])]
    (is (contains? (:forest state) :main))
    (is (contains? (get-in state [:forest :main]) "root"))))

(deftest test-expand-node
  (let [initial-state (make-initial-orchestrator-state [])
        candidates [{:candidate-action "Action 1" :prior-prob 0.5 :pragmatic-estimate 0.1 :epistemic-estimate 0.2 :confidence 1.0}
                    {:candidate-action "Action 2" :prior-prob 0.5 :pragmatic-estimate 0.3 :epistemic-estimate 0.4 :confidence 1.0}]
        expanded-state (expand-node initial-state :main "root" candidates)]
    (is (= 3 (count (get-in expanded-state [:forest :main]))))
    (is (contains? (get-in expanded-state [:forest :main]) "root-0"))
    (is (contains? (get-in expanded-state [:forest :main]) "root-1"))))

(deftest test-select-best-node
  (let [initial-state (make-initial-orchestrator-state [])
        candidates [{:candidate-action "Action 1" :prior-prob 0.5 :pragmatic-estimate 0.1 :epistemic-estimate 0.2 :confidence 1.0}
                    {:candidate-action "Action 2" :prior-prob 0.5 :pragmatic-estimate 0.3 :epistemic-estimate 0.4 :confidence 1.0}]
        expanded-state (expand-node initial-state :main "root" candidates)
        tree (get-in expanded-state [:forest :main])
        best-id (select-best-node tree "root" 1.0)]
    ;; Both children have 0 visits, so select-best-node (with -infinity for zero visits) should return one of them.
    (is (contains? #{"root-0" "root-1"} best-id))))

(deftest test-update-node-value
  (let [initial-state (make-initial-orchestrator-state [])
        tree (get-in initial-state [:forest :main])
        updated-tree (update-node-value tree "root" 0.1)]
    (is (= 1 (:visits (get updated-tree "root"))))
    ;; Initial visits was 0, so (1.0 * 0 + 0.1) / 1 = 0.1
    (is (= 0.1 (:value (get updated-tree "root"))))))

(deftest test-extract-best-policy
  (let [initial-state (make-initial-orchestrator-state [])
        candidates [{:candidate-action "Action 1" :prior-prob 0.5 :pragmatic-estimate 0.1 :epistemic-estimate 0.1 :confidence 1.0}
                    {:candidate-action "Action 2" :prior-prob 0.5 :pragmatic-estimate 0.8 :epistemic-estimate 0.8 :confidence 1.0}]
        expanded-state (expand-node initial-state :main "root" candidates)
        policy (extract-best-policy expanded-state :main)]
    (is (= ["Action 1"] policy))))

(deftest test-prune-branches
  (let [initial-state (make-initial-orchestrator-state [])
        candidates [{:candidate-action "Good" :prior-prob 0.5 :pragmatic-estimate 0.1 :epistemic-estimate 0.1 :confidence 1.0}
                    {:candidate-action "Bad" :prior-prob 0.5 :pragmatic-estimate 0.9 :epistemic-estimate 0.9 :confidence 1.0}]
        expanded-state (expand-node initial-state :main "root" candidates)
        pruned-state (prune-branches expanded-state :main 1.0)] ;; Bad has value 1.8
    (is (= 2 (count (get-in pruned-state [:forest :main]))))
    (is (contains? (get-in pruned-state [:forest :main]) "root-0"))
    (is (not (contains? (get-in pruned-state [:forest :main]) "root-1")))))
