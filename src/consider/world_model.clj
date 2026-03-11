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
  "Creates a transition function that uses a causal adjacency matrix (B-matrix).
   The adjacency matrix defines how states at t-1 influence states at t.
   Flatten all dimensions of all slots for matrix math."
  [adjacency-matrix slot-ids]
  (fn [internal-states action]
    ;; For now, we only use the adjacency matrix to influence transitions.
    (let [sorted-ids (sort slot-ids)
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
                                     new-pos (mapv #(n/entry next-state-vec (+ offset %)) (range dim))]
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
  "Merges multiple slots into a single slot (e.g., if they represent the same entity)."
  [belief-state target-id source-ids]
  (let [internal-states (:internal-states belief-state)
        target-slot (get internal-states target-id)
        ;; Simplified merge: just keep the target but remove the sources.
        ;; In a real implementation, we would average the means and combine variances.
        new-states (apply dissoc internal-states source-ids)]
    (assoc belief-state :internal-states new-states)))

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

(defn get-slot
  "Retrieves a slot from the internal states."
  [belief-state id]
  (get-in belief-state [:internal-states id]))

(defn remove-slot
  "Removes a slot from the internal states."
  [belief-state id]
  (update belief-state :internal-states dissoc id))

(defn validate-belief-state
  "Validates the belief state against its specification."
  [belief-state]
  (if (s/valid? ::wm-spec/belief-state belief-state)
    belief-state
    (throw (ex-info "Invalid belief state"
                    {:explain (s/explain-data ::wm-spec/belief-state belief-state)}))))
