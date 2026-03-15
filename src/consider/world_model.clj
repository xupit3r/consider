(ns consider.world-model
  "Implementation of the Probabilistic World Model (Active Inference)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.world-model :as wm-spec]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]))

(defn make-slot
  "Creates a new slot representing a hidden state."
  ([id position]
   (make-slot id position (vec (repeat (count position) 1.0))))
  ([id position variance]
   {:entity-id id
    :position position
    :variance variance}))

(defn make-belief-state
  "Constructs an initial belief state."
  ([]
   (make-belief-state {} []))
  ([internal-states preferences]
   {:internal-states internal-states
    :variational-free-energy 0.0
    :expected-free-energy 0.0
    :efe-components {:risk 0.0 :ambiguity 0.0}
    :preferences preferences
    :history []
    :history-limit 100}))

(defn with-generative-model
  "Attaches likelihood and transition dynamics to the belief state."
  [belief-state likelihood-fn transition-fn]
  (assoc belief-state
         :likelihood-mapping likelihood-fn
         :transition-dynamics transition-fn))

(defn predict-next-state
  "Uses transition dynamics to predict the next internal states given an action."
  [belief-state action]
  (if-let [transition-fn (:transition-dynamics belief-state)]
    (transition-fn (:internal-states belief-state) action)
    (:internal-states belief-state)))

(defn predict-observation
  "Uses likelihood mapping to predict observations from current internal states."
  [belief-state]
  (if-let [likelihood-fn (:likelihood-mapping belief-state)]
    (likelihood-fn (:internal-states belief-state))
    []))

(defn update-slot
  "Updates a slot in the internal states."
  [belief-state id position variance]
  (assoc-in belief-state [:internal-states id] (make-slot id position variance)))

(defn make-causal-transition-fn
  "Creates a transition function that supports interventions (do-calculus).
   The adjacency matrix defines natural evolution.
   The 'action' can now include an 'intervention' map {slot-id new-pos}."
  [adjacency-matrix slot-ids]
  (fn [internal-states action]
    (let [sorted-ids (sort slot-ids)
          ;; Extract intervention from action if present (e.g., {:type :do :target :me :value [1.0]})
          intervention (when (and (map? action) (= (:type action) :do))
                         action)

          state-data (mapcat #(:position (get internal-states %)) sorted-ids)
          d (count state-data)
          state-vec (native/dv d)]
      (dotimes [i d]
        (n/entry! state-vec i (double (nth state-data i))))

      (let [next-state-vec (n/mv adjacency-matrix state-vec)
            updated-states (loop [ids sorted-ids
                                  offset 0
                                  acc {}]
                             (if-let [id (first ids)]
                               (let [old-slot (get internal-states id)
                                     dim (count (:position old-slot))
                                     ;; If this slot is being intervened upon, use the intervention value
                                     new-pos (if (and intervention (= id (:target intervention)))
                                               (:value intervention)
                                               (mapv #(n/entry next-state-vec (+ offset %)) (range dim)))]
                                 (recur (rest ids)
                                        (+ offset dim)
                                        (assoc acc id (assoc old-slot :position new-pos))))
                               acc))]
        updated-states))))

(defn update-transition-dynamics
  "Updates the transition dynamics of the belief state using a new causal structure."
  [belief-state sparse-S]
  (let [slot-ids (keys (:internal-states belief-state))
        transition-fn (make-causal-transition-fn sparse-S slot-ids)]
    (assoc belief-state :transition-dynamics transition-fn)))

(defn grow-slots
  "Adds new hidden state slots to the internal world model."
  [belief-state new-slots]
  (update belief-state :internal-states merge
          (into {} (map (fn [s] [(:entity-id s) s]) new-slots))))

(defn merge-slots
  "Merges multiple slots into a target slot.
   Performs precision-weighted averaging of positions and variances."
  [belief-state target-id source-ids]
  (let [internal-states (:internal-states belief-state)
        target-slot (get internal-states target-id)
        source-slots (filter identity (map #(get internal-states %) source-ids))
        all-slots (conj source-slots target-slot)]
    (if (or (nil? target-slot) (empty? source-slots))
      belief-state
      (let [positions (map :position all-slots)
            variances (map :variance all-slots)
            dim (count (:position target-slot))

            ;; Precision-weighted average: pos = sum(pos_i / var_i) / sum(1 / var_i)
            ;; new_var = 1 / sum(1 / var_i)
            new-pos-and-var
            (mapv (fn [d]
                    (let [precisions (map #(/ 1.0 (max 1e-6 (nth % d))) variances)
                          sum-prec (reduce + precisions)
                          sum-weighted-pos (reduce + (map * (map #(nth % d) positions) precisions))]
                      [(/ sum-weighted-pos sum-prec) (/ 1.0 sum-prec)]))
                  (range dim))

            new-pos (mapv first new-pos-and-var)
            new-var (mapv second new-pos-and-var)

            updated-slot (assoc target-slot :position new-pos :variance new-var)

            new-states (-> (apply dissoc internal-states source-ids)
                           (assoc target-id updated-slot))]
        (assoc belief-state :internal-states new-states)))))

(defn dream-trajectory
  "Generates a simulated sequence of hidden states and observations (a dream).
   Uses the learned transition dynamics and current belief state as a seed."
  [belief-state steps]
  (let [transition-fn (:transition-dynamics belief-state)
        likelihood-fn (:likelihood-mapping belief-state)
        initial-states (:internal-states belief-state)]
    (loop [curr-states initial-states
           i 0
           acc []]
      (if (>= i steps)
        acc
        (let [;; In a dream, the agent explores potential actions (random or goal-directed)
              random-action (if (< (rand) 0.2) "DREAM_ACTION" "STAY")
              next-states (transition-fn curr-states random-action)
              observation (likelihood-fn next-states)
              entry {:internal-states next-states
                     :observation observation
                     ;; Dreams are highly uncertain, we can give them a 'surprise' value
                     :vfe 0.5}]
          (recur next-states (inc i) (conj acc entry)))))))

(defn identify-novel-entities
  "Identifies potential new hidden states (slots) based on residual prediction errors
   and observation dimension mismatches."
  [belief-state actual-obs predicted-obs]
  (let [actual-count (count actual-obs)
        predicted-count (count predicted-obs)
        threshold 100.0
        ;; 1. Handle surplus observations (extra objects)
        new-entities (if (> actual-count predicted-count)
                       (mapv (fn [obs]
                               (make-slot (keyword (str "entity-" (System/currentTimeMillis) "-" (rand-int 1000)))
                                          [obs]))
                             (subvec actual-obs predicted-count))
                       [])
        ;; 2. Handle large errors in existing observations
        residual-error-entities (let [common-count (min actual-count predicted-count)]
                                  (reduce (fn [acc i]
                                            (let [err (Math/abs (- (nth actual-obs i) (nth predicted-obs i)))]
                                              (if (> err threshold)
                                                (conj acc (make-slot (keyword (str "entity-residual-" (System/currentTimeMillis)))
                                                                     [err]))
                                                acc)))
                                          []
                                          (range common-count)))]
    (vec (concat new-entities residual-error-entities))))

(defn identify-redundant-slots
  "Identifies groups of slots that are highly similar and could be merged.
   Similarity is based on Euclidean distance between positions."
  [belief-state threshold]
  (let [internal-states (:internal-states belief-state)
        ids (vec (keys internal-states))]
    (loop [remaining-ids ids
           merges []]
      (if (empty? remaining-ids)
        merges
        (let [id (first remaining-ids)
              pos (:position (get internal-states id))
              others (rest remaining-ids)
              similar (filter (fn [oid]
                                (let [opos (:position (get internal-states oid))]
                                  (let [dist (Math/sqrt (reduce + (map (fn [a b] (Math/pow (- a b) 2)) pos opos)))]
                                    (< dist threshold))))
                              others)]
          (if (seq similar)
            (recur (vec (remove (set similar) others))
                   (conj merges {:target id :sources similar}))
            (recur (vec others)
                   merges)))))))

(defn get-slot
  "Retrieves a slot from the internal states."
  [belief-state id]
  (get-in belief-state [:internal-states id]))

(defn remove-slot
  "Removes a slot from the internal states."
  [belief-state id]
  (update belief-state :internal-states dissoc id))

;; --- Knowledge Foraging Support ---

(defn make-knowledge-slot
  "Creates a slot representing a knowledge domain cluster.
   Position = domain embedding (from graph structure).
   Variance = epistemic uncertainty (starts high, decreases as knowledge grows)."
  [domain-id embedding uncertainty]
  (make-slot domain-id embedding uncertainty))

(defn knowledge-likelihood-fn
  "Creates a likelihood function for knowledge foraging.
   Predicts expected observations given knowledge state.
   Observation vector: [n-new-entities, n-confirmed, n-contradictions, n-new-relations, topic-sim, quality]"
  [knowledge-graph]
  (fn [internal-states]
    (let [slots (vals internal-states)
          ;; Average position across slots gives expected knowledge state
          avg-pos (if (empty? slots)
                    [0.0 0.0 0.0 0.0 0.5 0.5]
                    (let [positions (map :position slots)
                          n (count positions)]
                      (mapv (fn [i]
                              (/ (reduce + (map #(nth % i 0.0) positions)) n))
                            (range 6))))]
      ;; Expected observation: moderate new entities, some confirmations
      [(max 0.0 (- 5.0 (nth avg-pos 0 0.0))) ;; fewer new entities as knowledge grows
       (min 10.0 (nth avg-pos 0 0.0)) ;; more confirmations
       0.0 ;; contradictions expected = 0
       (max 0.0 (- 10.0 (nth avg-pos 1 0.0))) ;; fewer new relations
       (nth avg-pos 4 0.5) ;; topic similarity stays
       0.6]))) ;; expected page quality

(defn knowledge-transition-fn
  "Creates a transition function for knowledge foraging.
   Models how visiting a URL changes the knowledge state."
  []
  (fn [internal-states action]
    ;; Knowledge accumulates: positions shift toward observed values
    (reduce-kv (fn [acc id slot]
                 (let [pos (:position slot)
                       var (:variance slot)
                       ;; Visiting a URL slightly reduces variance (uncertainty)
                       new-var (mapv #(max 0.01 (* % 0.95)) var)
                       ;; Position shifts slightly toward center (mean-reversion)
                       ;; Target center for knowledge foraging could be [0 0 0 0 1 1] (ideal similarity/quality)
                       center [0.0 0.0 0.0 0.0 0.5 0.5]
                       new-pos (mapv (fn [i] (+ (nth pos i) (* 0.01 (- (nth center i) (nth pos i))))) (range 6))]
                   (assoc acc id (assoc slot :position new-pos :variance new-var))))
               {}
               internal-states)))

(defn validate-belief-state
  "Validates the belief state against its specification."
  [belief-state]
  (if (s/valid? ::wm-spec/belief-state belief-state)
    belief-state
    (throw (ex-info "Invalid belief state"
                    {:explain (s/explain-data ::wm-spec/belief-state belief-state)}))))
