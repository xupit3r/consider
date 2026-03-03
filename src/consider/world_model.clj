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

(defn update-slot
  "Updates a slot in the internal states."
  [belief-state id position variance]
  (assoc-in belief-state [:internal-states id] (make-slot id position variance)))

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
