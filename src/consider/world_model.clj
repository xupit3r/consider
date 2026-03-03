(ns consider.world-model
  "Implementation of the Probabilistic World Model (Active Inference)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.world-model :as wm-spec]))

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
    :preferences preferences}))

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
