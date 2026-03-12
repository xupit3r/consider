(ns consider.web.knowledge
  "LLM-based knowledge extraction: text → entities + triples.
   Two-stage KGGen-style extraction via Ollama."
  (:require [consider.llm :as llm]
            [clojure.string :as str]
            [charred.api :as json]))

;; --- KnowledgeExtractor Protocol ---

(defprotocol KnowledgeExtractor
  "Protocol for extracting structured knowledge from text."
  (extract-entities [this text]
    "Extract entities with types from text. Returns [{:entity-name :entity-type}]")
  (extract-relations [this text entities]
    "Extract (subject, predicate, object) triples. Returns [{:subject :predicate :object}]")
  (canonicalize-entity [this entity existing-entities]
    "Returns the canonical name for an entity, merging synonyms.")
  (formulate-query [this gaps goals]
    "Generate a search query from knowledge gaps and goals. Returns a query string."))

;; --- Prompt Templates ---

(defn entity-extraction-prompt
  "Prompt for extracting entities from text."
  [text]
  (str "Extract all named entities from the following text. For each entity, provide its name and type.

Types include: Person, Organization, Concept, Location, Event, Technology, Theory, Field, Publication.

Text:
" (subs text 0 (min (count text) 3000)) "

Return ONLY a JSON array of objects with keys \"entity_name\" and \"entity_type\".
Example: [{\"entity_name\": \"Albert Einstein\", \"entity_type\": \"Person\"}]"))

(defn relation-extraction-prompt
  "Prompt for extracting relations between entities."
  [text entities]
  (let [entity-names (str/join ", " (map :entity-name entities))]
    (str "Given these entities: " entity-names "

Extract relationships (triples) from the following text. Each triple has a subject, predicate, and object.

Text:
" (subs text 0 (min (count text) 3000)) "

Return ONLY a JSON array of objects with keys \"subject\", \"predicate\", \"object\".
Example: [{\"subject\": \"Einstein\", \"predicate\": \"developed\", \"object\": \"General Relativity\"}]")))

(defn canonicalization-prompt
  "Prompt for entity canonicalization."
  [entity existing-names]
  (str "Given this entity name: \"" entity "\"

And these existing canonical names: " (str/join ", " (map #(str "\"" % "\"") existing-names)) "

If the entity is a synonym or abbreviation of an existing name, return that name.
Otherwise return the original entity name.

Return ONLY a JSON object with key \"canonical_name\"."))

(defn query-formulation-prompt
  "Prompt for generating search queries from knowledge gaps."
  [gaps goals]
  (str "You are an epistemic foraging agent. Generate a web search query to fill knowledge gaps.

Knowledge gaps (topics with high uncertainty):
" (str/join "\n" (map #(str "- " %) gaps)) "

Current goals:
" (str/join "\n" (map #(str "- " %) goals)) "

Generate a single focused search query that would help fill the most important knowledge gap.
Return ONLY a JSON object with key \"query\"."))

;; --- Parsing Helpers ---

(defn- parse-entities-response
  "Parses entity extraction LLM response."
  [raw-resp]
  (let [default [{:entity-name "Unknown" :entity-type "Concept"}]
        parsed (llm/robust-parse-json raw-resp default)]
    (if (sequential? parsed)
      (mapv (fn [e]
              {:entity-name (or (:entity_name e) (:entity-name e) "Unknown")
               :entity-type (or (:entity_type e) (:entity-type e) "Concept")})
            parsed)
      default)))

(defn- parse-triples-response
  "Parses relation extraction LLM response."
  [raw-resp]
  (let [default [{:subject "Unknown" :predicate "related-to" :object "Unknown"}]
        parsed (llm/robust-parse-json raw-resp default)]
    (if (sequential? parsed)
      (mapv (fn [t]
              {:subject (or (:subject t) "Unknown")
               :predicate (or (:predicate t) "related-to")
               :object (or (:object t) "Unknown")})
            parsed)
      default)))

;; --- Full Extraction Pipeline ---

(defn text-to-triples
  "Two-stage extraction: entities then relations.
   Takes a completion-fn that accepts a prompt string and returns a response string."
  [completion-fn text existing-entities]
  (let [;; Stage 1: Entity extraction
        entity-resp (completion-fn (entity-extraction-prompt text))
        entities (parse-entities-response entity-resp)
        ;; Merge with existing
        all-entities (into (vec existing-entities) entities)
        ;; Stage 2: Relation extraction
        relation-resp (completion-fn (relation-extraction-prompt text all-entities))
        triples (parse-triples-response relation-resp)]
    {:entities entities
     :triples triples}))

(defn formulate-search-query
  "Generates a search query from knowledge gaps and goals."
  [completion-fn gaps goals]
  (let [resp (completion-fn (query-formulation-prompt gaps goals))
        parsed (llm/robust-parse-json resp {:query (first (or (seq goals) ["general knowledge"]))})]
    (or (:query parsed) (first goals) "knowledge")))

(defn canonicalize-entities
  "Asks the LLM to pick a single canonical name from a group of potential synonyms.
   Returns a map with :old-names and :canonical-name."
  [completion-fn group all-existing-names]
  (let [prompt (str "Given this group of potential synonyms for the same entity: "
                    (str/join ", " (map #(str "\"" % "\"") group)) "\n\n"
                    "And these other existing entity names: "
                    (str/join ", " (take 20 (map #(str "\"" % "\"") all-existing-names))) "\n\n"
                    "Select the BEST canonical name to represent this group. "
                    "If one of the names in the group is already widely used or more descriptive, pick it. "
                    "Return ONLY a JSON object with key \"canonical_name\".")
        resp (completion-fn prompt)
        parsed (llm/robust-parse-json resp {:canonical_name (first group)})
        canonical (or (:canonical_name parsed) (first group))]
    (mapv (fn [old] {:old-name old :canonical-name canonical})
          (remove #(= % canonical) group))))

