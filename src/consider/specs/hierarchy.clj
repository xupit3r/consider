(ns consider.specs.hierarchy
  "Specifications for Hierarchical State Abstractions."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.world-model :as wm-spec]))

;; --- Level 2 (Conceptual) States ---

(s/def ::concept-id keyword?)

(s/def ::constituent-slots (s/coll-of keyword? :kind set?))

(s/def ::concept-slot
  (s/keys :req-un [::concept-id 
                 ::constituent-slots
                 ::wm-spec/position ;; Abstract center/state of the concept
                 ::wm-spec/variance]))

(s/def ::conceptual-states (s/map-of ::concept-id ::concept-slot))

;; --- Hierarchical Generative Model ---

(s/def ::hierarchy
  (s/keys :req-un [::conceptual-states]
          :opt-un [::transition-dynamics-l2]))

(s/def ::hierarchical-belief-state
  (s/merge ::wm-spec/belief-state
           (s/keys :req-un [::hierarchy])))
