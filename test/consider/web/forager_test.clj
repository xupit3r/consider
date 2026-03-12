(ns consider.web.forager-test
  (:require [clojure.test :refer :all]
            [consider.web.forager :as forager]
            [consider.web.crawler :as crawler]
            [consider.web.graph :as graph]))

(deftest test-forager-creation
  (testing "make-forager creates valid state"
    (let [f (forager/make-forager {})]
      (is (some? (:knowledge-graph f)))
      (is (some? (:crawler-state f)))
      (is (vector? (:knowledge-goals f)))
      (is (= 0 (:step-count f)))
      (is (map? (:stats f))))))

(deftest test-goal-management
  (testing "set-knowledge-goal with string"
    (let [f (forager/set-knowledge-goal (forager/make-forager {}) "active inference")]
      (is (= ["active inference"] (:knowledge-goals f)))))

  (testing "set-knowledge-goal with vector"
    (let [f (forager/set-knowledge-goal (forager/make-forager {})
                                         ["active inference" "free energy"])]
      (is (= ["active inference" "free energy"] (:knowledge-goals f))))))

(deftest test-seeding
  (testing "seed-from-url adds to frontier"
    (let [f (forager/seed-from-url (forager/make-forager {}) "https://example.com")]
      (is (= 1 (crawler/frontier-size (:crawler-state f))))))

  (testing "seed-from-wikipedia constructs Wikipedia URL"
    (let [f (forager/seed-from-wikipedia (forager/make-forager {}) "Active Inference")]
      (is (= 1 (crawler/frontier-size (:crawler-state f))))))

  (testing "seed-from-urls adds multiple URLs"
    (let [f (forager/seed-from-urls (forager/make-forager {})
                                     ["https://a.com" "https://b.com" "https://c.com"])]
      (is (= 3 (crawler/frontier-size (:crawler-state f)))))))

(deftest test-forager-stats
  (testing "forager-stats returns summary"
    (let [f (forager/make-forager {:knowledge-goals ["test"]})
          stats (forager/forager-stats f)]
      (is (= 0 (:step-count stats)))
      (is (= ["test"] (:knowledge-goals stats)))
      (is (= 0 (:frontier-size stats)))
      (is (= 0 (:visited-count stats)))
      (is (map? (:graph-stats stats))))))

(deftest test-forager-observation
  (testing "forager->observation returns 6-dim vector"
    (let [f (forager/make-forager {})
          obs (forager/forager->observation f)]
      (is (= 6 (count obs)))
      (is (every? number? obs)))))

(deftest test-sleep-consolidate
  (testing "sleep-consolidate finds gaps and returns updated state"
    (let [f (forager/make-forager {:knowledge-goals ["test"]})]
      ;; Add some entities to the graph
      (graph/transact-entity! (:knowledge-graph f) {:entity-name "A" :entity-type "Concept"})
      (graph/transact-entity! (:knowledge-graph f) {:entity-name "B" :entity-type "Concept"})
      (let [consolidated (forager/sleep-consolidate f)]
        (is (some? (:last-sleep-gaps consolidated)))))))

(deftest test-forage-step-empty-frontier
  (testing "forage-step with empty frontier returns nil observation"
    (let [f (forager/make-forager {})
          [updated-f obs] (forager/forage-step f)]
      (is (nil? obs))
      (is (some? updated-f)))))
