(ns consider.web.graph-test
  (:require [clojure.test :refer :all]
            [consider.web.graph :as graph]))

(deftest test-knowledge-graph-creation
  (testing "creates a knowledge graph"
    (let [kg (graph/make-knowledge-graph)]
      (is (some? (:uri kg)))
      (is (some? (:connection kg)))
      (is (= 0 @(:entity-count kg)))
      (is (= 0 @(:triple-count kg))))))

(deftest test-entity-operations
  (let [kg (graph/make-knowledge-graph)]
    (testing "transact entity"
      (let [result (graph/transact-entity! kg {:entity-name "Active Inference"
                                               :entity-type "Theory"})]
        (is (= "Active Inference" (:entity/name result)))
        (is (= 1 @(:entity-count kg)))))

    (testing "transact multiple entities"
      (graph/transact-entity! kg {:entity-name "Karl Friston" :entity-type "Person"})
      (graph/transact-entity! kg {:entity-name "UCL" :entity-type "Organization"})
      (is (= 3 @(:entity-count kg))))

    (testing "query all entities"
      (let [entities (graph/query-all-entities kg)]
        (is (>= (count entities) 3))
        (is (some #(= "Active Inference" (first %)) entities))
        (is (some #(= "Karl Friston" (first %)) entities))))))

(deftest test-triple-operations
  (let [kg (graph/make-knowledge-graph)]
    ;; Set up entities
    (graph/transact-entity! kg {:entity-name "Karl Friston" :entity-type "Person"})
    (graph/transact-entity! kg {:entity-name "Active Inference" :entity-type "Theory"})
    (graph/transact-entity! kg {:entity-name "Free Energy Principle" :entity-type "Theory"})

    (testing "transact triple"
      (let [result (graph/transact-triple! kg {:subject "Karl Friston"
                                               :predicate "developed"
                                               :object "Active Inference"})]
        (is (= "Karl Friston" (:triple/subject result)))
        (is (= 1 @(:triple-count kg)))))

    (testing "transact multiple triples"
      (graph/transact-triple! kg {:subject "Active Inference"
                                  :predicate "based-on"
                                  :object "Free Energy Principle"})
      (graph/transact-triple! kg {:subject "Karl Friston"
                                  :predicate "works-at"
                                  :object "UCL"})
      (is (= 3 @(:triple-count kg))))

    (testing "query all triples"
      (let [triples (graph/query-all-triples kg)]
        (is (>= (count triples) 3))
        (is (some #(= "developed" (second %)) triples))))

    (testing "query neighbors"
      (let [neighbors (graph/query-neighbors kg "Karl Friston")]
        (is (>= (count (:outgoing neighbors)) 1))
        (is (some #(= "developed" (first %)) (:outgoing neighbors)))))

    (testing "query triples about entity"
      (let [triples (graph/query-triples-about kg "Active Inference")]
        (is (>= (count triples) 1))))))

(deftest test-extraction-batch
  (let [kg (graph/make-knowledge-graph)
        extraction {:entities [{:entity-name "E1" :entity-type "Concept"}
                               {:entity-name "E2" :entity-type "Concept"}]
                    :triples [{:subject "E1" :predicate "relates-to" :object "E2"}]}]
    (testing "transact-extraction! stores both entities and triples"
      (let [result (graph/transact-extraction! kg extraction :source-url "https://example.com")]
        (is (= 2 (:entities-added result)))
        (is (= 1 (:triples-added result)))
        (is (= 2 @(:entity-count kg)))
        (is (= 1 @(:triple-count kg)))))))

(deftest test-knowledge-gaps
  (let [kg (graph/make-knowledge-graph)]
    ;; Create a graph where some entities are well-connected and others aren't
    (graph/transact-entity! kg {:entity-name "Hub" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "Connected1" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "Connected2" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "Isolated" :entity-type "Concept"})

    (graph/transact-triple! kg {:subject "Hub" :predicate "links-to" :object "Connected1"})
    (graph/transact-triple! kg {:subject "Hub" :predicate "links-to" :object "Connected2"})
    (graph/transact-triple! kg {:subject "Connected1" :predicate "links-to" :object "Hub"})

    (testing "find-knowledge-gaps identifies poorly-connected entities"
      (let [gaps (graph/find-knowledge-gaps kg)]
        (is (vector? gaps))
        ;; Isolated should have 0 connections and appear first
        (is (= "Isolated" (:entity-name (first gaps))))
        (is (zero? (:connection-count (first gaps))))))

    (testing "find-sparse-regions returns entities below threshold"
      (let [sparse (graph/find-sparse-regions kg 0)]
        (is (some #(= "Isolated" (:entity-name %)) sparse))))))

(deftest test-graph-stats
  (let [kg (graph/make-knowledge-graph)]
    (graph/transact-entity! kg {:entity-name "A" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "B" :entity-type "Concept"})
    (graph/transact-triple! kg {:subject "A" :predicate "r" :object "B"})

    (testing "graph-stats returns counts"
      (let [stats (graph/graph-stats kg)]
        (is (= 2 (:entity-count stats)))
        (is (= 1 (:triple-count stats)))))))

(deftest test-knowledge-embedding
  (let [kg (graph/make-knowledge-graph)]
    (graph/transact-entity! kg {:entity-name "A" :entity-type "Concept"})
    (graph/transact-entity! kg {:entity-name "B" :entity-type "Concept"})
    (graph/transact-triple! kg {:subject "A" :predicate "r" :object "B"})

    (testing "knowledge-embedding returns 6-dim vector"
      (let [emb (graph/knowledge-embedding kg)]
        (is (= 6 (count emb)))
        (is (every? number? emb))
        ;; First dim = entity count
        (is (>= (first emb) 2.0))))))

(deftest test-merge-entities
  (testing "7d: Test merge-entities!"
    (let [kg (graph/make-knowledge-graph)]
      (graph/transact-entity! kg {:entity-name "AI" :entity-type "Concept"})
      (graph/transact-entity! kg {:entity-name "Artificial Intelligence" :entity-type "Concept"})
      (graph/transact-triple! kg {:subject "AI" :predicate "is-a" :object "Field"})
      (graph/transact-triple! kg {:subject "Researchers" :predicate "study" :object "AI"})
      (let [result (graph/merge-entities! kg "AI" "Artificial Intelligence")]
        (is (= 2 (:triples-rewritten result)))
        ;; Verify new triples reference canonical name
        (let [triples (graph/query-triples-about kg "Artificial Intelligence")]
          (is (>= (count triples) 2)))))))

(deftest test-100-triples
  (testing "store and query 100 triples"
    (let [kg (graph/make-knowledge-graph)]
      ;; Create 20 entities
      (doseq [i (range 20)]
        (graph/transact-entity! kg {:entity-name (str "Entity-" i) :entity-type "Concept"}))

      ;; Create 100 triples
      (doseq [i (range 100)]
        (graph/transact-triple! kg {:subject (str "Entity-" (mod i 20))
                                    :predicate (str "relation-" (mod i 5))
                                    :object (str "Entity-" (mod (inc i) 20))}))

      (is (= 20 @(:entity-count kg)))
      (is (= 100 @(:triple-count kg)))

      ;; Query subgraph
      (let [neighbors (graph/query-neighbors kg "Entity-0")]
        (is (pos? (+ (count (:outgoing neighbors))
                     (count (:incoming neighbors))))))

      ;; Detect gaps
      (let [gaps (graph/find-knowledge-gaps kg)]
        (is (vector? gaps))
        (is (= 20 (count gaps)))))))
