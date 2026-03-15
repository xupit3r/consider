(ns consider.web.graph
  "Knowledge graph storage using Asami. Provides entity/triple storage,
   Datalog querying, gap detection, and graph-to-vector embeddings."
  (:require [asami.core :as d]
            [clojure.string :as str]))

;; --- Graph Creation ---

(defn make-knowledge-graph
  "Creates a new in-memory Asami knowledge graph."
  ([]
   (Thread/sleep 1)
   (make-knowledge-graph (str "asami:mem://consider-kg-" (System/currentTimeMillis))))
  ([uri]
   (d/create-database uri)
   {:uri uri
    :connection (d/connect uri)
    :entity-count (atom 0)
    :triple-count (atom 0)}))

;; --- Entity Operations ---

(defn transact-entity!
  "Stores an entity in the knowledge graph.
   Entity: {:entity-name string :entity-type string}"
  [kg entity & {:keys [source-url]}]
  (let [conn (:connection kg)
        entity-data (cond-> {:id (keyword "entity" (str/replace (:entity-name entity) #"\s+" "-"))
                             :entity/name (:entity-name entity)
                             :entity/type (:entity-type entity)
                             :entity/confirmed-count 1}
                      source-url (assoc :entity/source-url source-url))]
    @(d/transact conn {:tx-data [entity-data]})
    (swap! (:entity-count kg) inc)
    entity-data))

(defn transact-triple!
  "Stores a triple (subject, predicate, object) in the knowledge graph."
  [kg triple & {:keys [source-url confidence]}]
  (let [conn (:connection kg)
        triple-data (cond-> {:id (keyword "triple"
                                          (str (str/replace (:subject triple) #"\s+" "-")
                                               "-" (str/replace (:predicate triple) #"\s+" "-")
                                               "-" (str/replace (:object triple) #"\s+" "-")))
                             :triple/subject (:subject triple)
                             :triple/predicate (:predicate triple)
                             :triple/object (:object triple)
                             :triple/confidence (or confidence 1.0)}
                      source-url (assoc :triple/source-url source-url))]
    @(d/transact conn {:tx-data [triple-data]})
    (swap! (:triple-count kg) inc)
    triple-data))

(defn transact-extraction!
  "Stores a full extraction result (entities + triples) in the graph.
   Optimized to use a single transaction."
  [kg extraction-result & {:keys [source-url]}]
  (let [{:keys [entities triples]} extraction-result
        conn (:connection kg)
        entity-txs (mapv (fn [entity]
                           (cond-> {:id (keyword "entity" (str/replace (:entity-name entity) #"\s+" "-"))
                                    :entity/name (:entity-name entity)
                                    :entity/type (:entity-type entity)
                                    :entity/confirmed-count 1}
                             source-url (assoc :entity/source-url source-url)))
                         entities)
        triple-txs (mapv (fn [triple]
                           (cond-> {:id (keyword "triple"
                                                 (str (str/replace (:subject triple) #"\s+" "-")
                                                      "-" (str/replace (:predicate triple) #"\s+" "-")
                                                      "-" (str/replace (:object triple) #"\s+" "-")))
                                    :triple/subject (:subject triple)
                                    :triple/predicate (:predicate triple)
                                    :triple/object (:object triple)
                                    :triple/confidence 1.0}
                             source-url (assoc :triple/source-url source-url)))
                         triples)
        tx-data (vec (concat entity-txs triple-txs))]
    (when (seq tx-data)
      @(d/transact conn {:tx-data tx-data})
      (swap! (:entity-count kg) + (count entities))
      (swap! (:triple-count kg) + (count triples)))
    {:entities-added (count entities)
     :triples-added (count triples)}))

;; --- Querying ---

(defn query-all-entities
  "Returns all entities in the graph."
  [kg]
  (let [db (d/db (:connection kg))]
    (d/q '[:find ?name ?type
           :where
           [?e :entity/name ?name]
           [?e :entity/type ?type]]
         db)))

(defn query-entity
  "Queries for a specific entity by name."
  [kg entity-name]
  (let [db (d/db (:connection kg))]
    (d/q '[:find ?name ?type
           :in $ ?target-name
           :where
           [?e :entity/name ?name]
           [?e :entity/type ?type]
           [(= ?name ?target-name)]]
         db entity-name)))

(defn query-neighbors
  "Finds all entities connected to a given entity via triples."
  [kg entity-name]
  (let [db (d/db (:connection kg))
        as-subject (d/q '[:find ?pred ?obj
                          :in $ ?name
                          :where
                          [?t :triple/subject ?subj]
                          [?t :triple/predicate ?pred]
                          [?t :triple/object ?obj]
                          [(= ?subj ?name)]]
                        db entity-name)
        as-object (d/q '[:find ?pred ?subj
                         :in $ ?name
                         :where
                         [?t :triple/subject ?subj]
                         [?t :triple/predicate ?pred]
                         [?t :triple/object ?obj]
                         [(= ?obj ?name)]]
                       db entity-name)]
    {:outgoing (vec as-subject)
     :incoming (vec as-object)}))

(defn query-all-triples
  "Returns all triples in the graph."
  [kg]
  (let [db (d/db (:connection kg))]
    (d/q '[:find ?subj ?pred ?obj
           :where
           [?t :triple/subject ?subj]
           [?t :triple/predicate ?pred]
           [?t :triple/object ?obj]]
         db)))

(defn query-triples-about
  "Returns all triples involving a given entity (as subject or object)."
  [kg entity-name]
  (let [db (d/db (:connection kg))]
    (into
     (vec (d/q '[:find ?subj ?pred ?obj
                 :in $ ?name
                 :where
                 [?t :triple/subject ?subj]
                 [?t :triple/predicate ?pred]
                 [?t :triple/object ?obj]
                 [(= ?subj ?name)]]
               db entity-name))
     (d/q '[:find ?subj ?pred ?obj
            :in $ ?name
            :where
            [?t :triple/subject ?subj]
            [?t :triple/predicate ?pred]
            [?t :triple/object ?obj]
            [(= ?obj ?name)]]
          db entity-name))))

;; --- Knowledge Gap Detection ---

(defn find-knowledge-gaps
  "Identifies entities with few connections — likely knowledge gaps.
   Returns entities sorted by connection count (ascending = least known)."
  [kg]
  (let [entities (query-all-entities kg)
        entity-names (map first entities)]
    (->> entity-names
         (map (fn [name]
                (let [neighbors (query-neighbors kg name)
                      connection-count (+ (count (:outgoing neighbors))
                                          (count (:incoming neighbors)))]
                  {:entity-name name
                   :connection-count connection-count})))
         (sort-by :connection-count)
         vec)))

(defn find-sparse-regions
  "Finds clusters of entities that are poorly connected to each other.
   Returns entity names that appear in triples but have few cross-connections."
  [kg min-gap-threshold]
  (let [gaps (find-knowledge-gaps kg)]
    (filterv #(<= (:connection-count %) min-gap-threshold) gaps)))

;; --- Graph Statistics ---

(defn graph-stats
  "Returns statistics about the knowledge graph."
  [kg]
  {:entity-count @(:entity-count kg)
   :triple-count @(:triple-count kg)
   :unique-entities (count (query-all-entities kg))
   :unique-triples (count (query-all-triples kg))})

;; --- Knowledge Embedding ---

(defn knowledge-embedding
  "Derives a fixed-dimensional embedding from the knowledge graph structure.
   Returns a vector suitable for use as slot position in active inference.

   Dimensions: [total-entities, total-triples, avg-connectivity,
                max-connectivity, isolation-ratio, graph-density]"
  [kg]
  (let [stats (graph-stats kg)
        gaps (find-knowledge-gaps kg)
        n-entities (max 1 (:unique-entities stats))
        n-triples (:unique-triples stats)
        connectivities (map :connection-count gaps)
        avg-conn (if (seq connectivities)
                   (/ (double (reduce + connectivities)) (count connectivities))
                   0.0)
        max-conn (if (seq connectivities)
                   (double (apply max connectivities))
                   0.0)
        isolated (count (filter #(zero? (:connection-count %)) gaps))
        isolation-ratio (/ (double isolated) n-entities)
        ;; Graph density: actual edges / possible edges
        max-edges (* n-entities (dec n-entities))
        density (if (pos? max-edges)
                  (/ (double n-triples) max-edges)
                  0.0)]
    [(double n-entities)
     (double n-triples)
     avg-conn
     max-conn
     isolation-ratio
     density]))

;; --- Entity Merging (Sleep Phase) ---

(defn- retract-entity!
  "Helper: Retracts all datoms associated with an entity ID (as subject or object)."
  [conn eid]
  (let [db (d/db conn)
        ;; Outgoing
        out (d/q '[:find ?p ?v :in $ ?e :where [?e ?p ?v]] db eid)
        ;; Incoming
        in (d/q '[:find ?s ?p :in $ ?e :where [?s ?p ?e]] db eid)
        tx-data (concat (map (fn [[p v]] [:db/retract eid p v]) out)
                        (map (fn [[s p]] [:db/retract s p eid]) in))]
    (when (seq tx-data)
      @(d/transact conn {:tx-data (vec tx-data)}))))

(defn merge-entities!
  "Merges two entities in the knowledge graph (for sleep-phase consolidation).
   Rewrites all triples referencing old-name to use canonical-name."
  [kg old-name canonical-name]
  (let [conn (:connection kg)
        db (d/db conn)
        ;; Find triples with old name as subject
        subj-triples (d/q '[:find ?t ?pred ?obj
                            :in $ ?old
                            :where
                            [?t :triple/subject ?old]
                            [?t :triple/predicate ?pred]
                            [?t :triple/object ?obj]]
                          db old-name)
        ;; Find triples with old name as object
        obj-triples (d/q '[:find ?t ?subj ?pred
                           :in $ ?old
                           :where
                           [?t :triple/subject ?subj]
                           [?t :triple/predicate ?pred]
                           [?t :triple/object ?old]]
                         db old-name)
        old-entity-id (keyword "entity" (str/replace old-name #"\s+" "-"))]

    ;; 1. Create new triples with canonical name
    (doseq [[t-id pred obj] subj-triples]
      (transact-triple! kg {:subject canonical-name :predicate pred :object obj}))
    (doseq [[t-id subj pred] obj-triples]
      (transact-triple! kg {:subject subj :predicate pred :object canonical-name}))

    ;; 2. Retract old triples (entities)
    (doseq [[t-id & _] (concat subj-triples obj-triples)]
      (retract-entity! conn t-id))

    ;; 3. Retract old entity itself
    (retract-entity! conn old-entity-id)

    {:merged old-name :into canonical-name
     :triples-rewritten (+ (count subj-triples) (count obj-triples))}))

(defn decay-confidence!
  "Sleep phase: decays confidence of triples.
   Triples seen from multiple sources could stay high (requires source tracking)."
  [kg decay-rate threshold]
  (let [conn (:connection kg)
        db (d/db conn)
        triples (d/q '[:find ?t ?conf
                       :where
                       [?t :triple/confidence ?conf]]
                     db)]
    (doseq [[t-id conf] triples]
      (let [new-conf (* conf (- 1.0 decay-rate))]
        @(d/transact conn {:tx-data [{:id t-id :triple/confidence new-conf}]})))
    kg))
