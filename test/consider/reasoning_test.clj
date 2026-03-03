(ns consider.reasoning-test
  (:require [clojure.test :refer :all]
            [consider.executive :refer :all]
            [consider.llm :as llm]
            [clojure.spec.alpha :as s]))

(deftest test-reason-loop
  (let [initial-state (make-initial-orchestrator-state [])
        mock-llm (llm/make-mock-llm)
        final-state (reason initial-state mock-llm mock-llm :main 5 1.0)]
    (testing "Reasoning loop expands the tree"
      (let [tree (get-in final-state [:forest :main])]
        (is (> (count tree) 1))
        (is (> (:visits (get tree "root")) 0))))))
