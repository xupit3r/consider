(ns consider.web.knowledge-test
  (:require [clojure.test :refer :all]
            [consider.web.knowledge :as knowledge]
            [charred.api :as json]))

(defn mock-completion-fn
  "A mock LLM that returns predetermined entity/relation extractions."
  [prompt]
  (cond
    ;; Entity extraction prompt
    (clojure.string/includes? prompt "Extract all named entities")
    (json/write-json-str
     [{:entity_name "Active Inference" :entity_type "Theory"}
      {:entity_name "Karl Friston" :entity_type "Person"}
      {:entity_name "Free Energy Principle" :entity_type "Theory"}
      {:entity_name "UCL" :entity_type "Organization"}])

    ;; Relation extraction prompt
    (clojure.string/includes? prompt "Extract relationships")
    (json/write-json-str
     [{:subject "Karl Friston" :predicate "developed" :object "Active Inference"}
      {:subject "Active Inference" :predicate "based-on" :object "Free Energy Principle"}
      {:subject "Karl Friston" :predicate "works-at" :object "UCL"}])

    ;; Query formulation prompt
    (clojure.string/includes? prompt "Generate a web search query")
    (json/write-json-str {:query "variational free energy neuroscience applications"})

    ;; Canonicalization
    (clojure.string/includes? prompt "synonym or abbreviation")
    (json/write-json-str {:canonical_name "Active Inference"})

    :else
    "{}"))

(deftest test-text-to-triples
  (testing "extracts entities and triples"
    (let [text "Active inference was developed by Karl Friston at UCL based on the Free Energy Principle."
          result (knowledge/text-to-triples mock-completion-fn text [])]
      (is (map? result))
      (is (vector? (:entities result)))
      (is (vector? (:triples result)))
      (is (>= (count (:entities result)) 3))
      (is (>= (count (:triples result)) 2))
      ;; Check entity structure
      (is (every? #(contains? % :entity-name) (:entities result)))
      (is (every? #(contains? % :entity-type) (:entities result)))
      ;; Check triple structure
      (is (every? #(contains? % :subject) (:triples result)))
      (is (every? #(contains? % :predicate) (:triples result)))
      (is (every? #(contains? % :object) (:triples result))))))

(deftest test-text-to-triples-with-existing
  (testing "merges with existing entities"
    (let [existing [{:entity-name "Karl Friston" :entity-type "Person"}]
          result (knowledge/text-to-triples mock-completion-fn "Some text" existing)]
      ;; Should include existing entities in the extraction context
      (is (sequential? (:entities result))))))

(deftest test-formulate-search-query
  (testing "generates query from gaps and goals"
    (let [query (knowledge/formulate-search-query
                 mock-completion-fn
                 ["Variational Bayes" "Markov Blankets"]
                 ["Learn about active inference"])]
      (is (string? query))
      (is (pos? (count query))))))

(deftest test-prompt-templates
  (testing "entity extraction prompt includes text"
    (let [prompt (knowledge/entity-extraction-prompt "Hello world")]
      (is (clojure.string/includes? prompt "Hello world"))
      (is (clojure.string/includes? prompt "entity_name"))))

  (testing "relation extraction prompt includes entities"
    (let [prompt (knowledge/relation-extraction-prompt "text"
                  [{:entity-name "A"} {:entity-name "B"}])]
      (is (clojure.string/includes? prompt "A, B"))
      (is (clojure.string/includes? prompt "subject"))))

  (testing "query formulation prompt includes gaps and goals"
    (let [prompt (knowledge/query-formulation-prompt ["gap1"] ["goal1"])]
      (is (clojure.string/includes? prompt "gap1"))
      (is (clojure.string/includes? prompt "goal1")))))
