(ns consider.specs.web
  "Specifications for web foraging data structures."
  (:require [clojure.spec.alpha :as s]))

;; --- Crawler ---

(s/def ::url string?)
(s/def ::anchor-text string?)
(s/def ::context string?)
(s/def ::link (s/keys :req-un [::url] :opt-un [::anchor-text ::context]))
(s/def ::links (s/coll-of ::link))

(s/def ::visited (s/coll-of ::url :kind set?))
(s/def ::frontier-entry (s/keys :req-un [::url ::efe-score]))
(s/def ::efe-score number?)

(s/def ::user-agent string?)
(s/def ::default-delay-ms pos-int?)
(s/def ::max-per-domain pos-int?)
(s/def ::crawler-config (s/keys :req-un [::user-agent ::default-delay-ms ::max-per-domain]))

(s/def ::last-ms number?)
(s/def ::delay-ms pos-int?)
(s/def ::rate-limiter (s/keys :req-un [::last-ms ::delay-ms]))
(s/def ::limiters (s/map-of string? ::rate-limiter))

(s/def ::robots-rules map?)
(s/def ::robots (s/map-of string? ::robots-rules))

(s/def ::crawler-state
  (s/keys :req-un [::visited ::frontier ::robots ::limiters ::crawler-config]))

;; --- Extractor ---

(s/def ::title (s/nilable string?))
(s/def ::text string?)
(s/def ::chunks (s/coll-of string?))
(s/def ::extracted-content (s/keys :req-un [::title ::text ::links ::chunks]))

;; --- Knowledge ---

(s/def ::entity-name string?)
(s/def ::entity-type string?)
(s/def ::entity (s/keys :req-un [::entity-name ::entity-type]))
(s/def ::entities (s/coll-of ::entity))

(s/def ::subject string?)
(s/def ::predicate string?)
(s/def ::object string?)
(s/def ::triple (s/keys :req-un [::subject ::predicate ::object]))
(s/def ::triples (s/coll-of ::triple))

(s/def ::extraction-result (s/keys :req-un [::entities ::triples]))

;; --- Knowledge Graph ---

(s/def ::graph-uri string?)
(s/def ::knowledge-graph (s/keys :req-un [::graph-uri]))

;; --- Frontier ---

(s/def ::frontier (s/coll-of ::frontier-entry))

;; --- Forager ---

(s/def ::knowledge-goals (s/coll-of string?))
(s/def ::forager-config
  (s/keys :req-un [::crawler-config ::knowledge-goals]))

(s/def ::step-count nat-int?)
(s/def ::forager-state
  (s/keys :req-un [::crawler-state ::step-count]))

;; --- Observation Vector ---
;; Fixed-dim vector compatible with existing VFE/EFE:
;; [n-new-entities, n-confirmed-entities, n-contradictions, n-new-relations, topic-similarity-to-goal, page-quality]
(s/def ::observation-vector (s/coll-of number? :kind vector? :count 6))
