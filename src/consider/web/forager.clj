(ns consider.web.forager
  "Epistemic Foraging Loop Orchestrator. Connects web crawling modules to the
   Active Inference core. Implements curiosity-driven and goal-directed foraging."
  (:require [consider.web.crawler :as crawler]
            [consider.web.extractor :as extractor]
            [consider.web.knowledge :as knowledge]
            [consider.web.graph :as graph]
            [consider.world-model :as wm]
            [consider.inference :as inf]
            [consider.core :as core]
            [consider.llm :as llm]
            [clojure.string :as str]
            [clojure.set :as set]))

;; --- Forager State ---

(defn make-forager
  "Creates a new forager state.
   Options:
     :knowledge-goals - vector of topic strings the agent wants to learn about
     :crawler-opts - options passed to make-crawler
     :llm-completion-fn - function that takes a prompt string, returns response string
     :mode - :curiosity or :goal-directed (default: :curiosity)"
  [opts]
  (let [kg (graph/make-knowledge-graph)
        crawler-state (crawler/make-crawler (or (:crawler-opts opts) {}))]
    {:knowledge-graph kg
     :crawler-state crawler-state
     :knowledge-goals (or (:knowledge-goals opts) [])
     :llm-completion-fn (:llm-completion-fn opts)
     :mode (or (:mode opts) :curiosity)
     :step-count 0
     :stats {:pages-fetched 0
             :entities-extracted 0
             :triples-extracted 0
             :failed-fetches 0}}))

;; --- Observation Vector Construction ---

(defn- build-observation-vector
  "Constructs the fixed-dim observation vector from extraction results.
   [n-new-entities, n-confirmed, n-contradictions, n-new-relations, topic-similarity, page-quality]"
  [extraction-result kg goals page-text]
  (let [{:keys [entities triples]} extraction-result
        existing-entities (graph/query-all-entities kg)
        existing-names (set (map first existing-entities))
        new-ents (count (remove #(contains? existing-names (:entity-name %)) entities))
        confirmed (count (filter #(contains? existing-names (:entity-name %)) entities))
        ;; Simple contradiction detection: triples that contradict existing ones
        contradictions 0
        new-rels (count triples)
        ;; Topic similarity: fraction of goal terms found in page text
        goal-terms (when (seq goals)
                     (mapcat #(str/split (str/lower-case %) #"\s+") goals))
        lower-text (str/lower-case (or page-text ""))
        topic-sim (if (seq goal-terms)
                    (let [matches (count (filter #(str/includes? lower-text %) goal-terms))]
                      (/ (double matches) (count goal-terms)))
                    0.5)
        ;; Page quality: simple heuristic based on text length and entity density
        text-len (count (or page-text ""))
        quality (cond
                  (< text-len 100) 0.1
                  (< text-len 500) 0.3
                  (< text-len 2000) 0.6
                  :else (min 1.0 (/ (double (+ new-ents (count triples))) 10.0)))]
    [(double new-ents)
     (double confirmed)
     (double contradictions)
     (double new-rels)
     (double topic-sim)
     (double quality)]))

;; --- EFE Scoring for URLs ---

(defn- score-url
  "Scores a URL for the crawl frontier using EFE-like heuristics.
   Lower score = higher priority (we minimize EFE)."
  [link goals kg mode]
  (let [{:keys [url anchor-text context]} link
        ;; Risk: distance from goal topics
        goal-terms (when (seq goals)
                     (mapcat #(str/split (str/lower-case %) #"\s+") goals))
        link-text (str/lower-case (str (or anchor-text "") " " (or context "")))
        goal-match (if (seq goal-terms)
                     (let [matches (count (filter #(str/includes? link-text %) goal-terms))]
                       (/ (double matches) (count goal-terms)))
                     0.0)
        risk (- 1.0 goal-match)
        ;; Ambiguity: how unknown is this link's domain?
        domain (crawler/extract-domain url)
        ;; Unknown domains get higher ambiguity (= more epistemic value)
        gaps (graph/find-knowledge-gaps kg)
        gap-names (set (map :entity-name gaps))
        ;; Check if anchor text mentions any gap entities
        mentions-gap (some #(str/includes? link-text (str/lower-case %)) gap-names)
        ambiguity (if mentions-gap 0.3 0.7)

        ;; Weighting based on mode
        [w-risk w-amb] (if (= mode :goal-directed)
                         [0.8 0.2]
                         [0.3 0.7])]
    ;; EFE = risk + ambiguity (lower = better)
    (+ (* w-risk risk) (* w-amb ambiguity))))

;; --- Core Foraging Step ---

(defn forage-step
  "Performs a single foraging step:
   1. Pick next URL from frontier (or use provided URL)
   2. Fetch page
   3. Extract content
   4. Extract knowledge via LLM
   5. Store in graph
   6. Score and enqueue new links
   7. Build observation vector

   Returns [updated-forager-state observation-vector] or [state nil] on failure."
  [forager-state & {:keys [url search-engine]}]
  (let [{:keys [crawler-state knowledge-graph knowledge-goals llm-completion-fn mode]} forager-state
        ;; 1. Get URL
        [crawler-after-pick target-url] (if url
                                          [crawler-state url]
                                          (crawler/next-url crawler-state))]
    (if-not target-url
      [(assoc forager-state :crawler-state crawler-after-pick) nil]
      ;; 2. Fetch page
      (let [[crawler-after-fetch page-result] (crawler/fetch-page crawler-after-pick target-url)]
        (if (or (nil? page-result) (nil? (:body page-result)) (not= 200 (:status page-result)))
          ;; Fetch failed
          [(-> forager-state
               (assoc :crawler-state crawler-after-fetch)
               (update-in [:stats :failed-fetches] inc)
               (update :step-count inc))
           nil]
          ;; 3. Extract content
          (let [content (if search-engine
                          {:links (extractor/extract-search-results (:body page-result) search-engine target-url)
                           :text "" :chunks []}
                          (extractor/extract-content (:body page-result) target-url))
                text (:text content)
                links (:links content)
                chunks (:chunks content)]

            ;; 4. Extract knowledge from each chunk
            (let [extraction
                  (cond
                    (nil? llm-completion-fn)
                    {:entities (mapv (fn [l]
                                       {:entity-name (or (:title l) (:anchor-text l) (:url l))
                                        :entity-type (if search-engine "SearchResult" "WebPage")})
                                     (take 10 links))
                     :triples []}

                    (satisfies? llm/KnowledgeExtractor llm-completion-fn)
                    (llm/extract-knowledge llm-completion-fn text [])

                    (fn? llm-completion-fn)
                    (let [existing-ents (map (fn [[n t]] {:entity-name n :entity-type t})
                                             (graph/query-all-entities knowledge-graph))]
                      (reduce (fn [acc chunk]
                                (let [result (knowledge/text-to-triples llm-completion-fn chunk
                                                                        (:entities acc))]
                                  {:entities (into (:entities acc) (:entities result))
                                   :triples (into (:triples acc) (:triples result))}))
                              {:entities (vec existing-ents) :triples []}
                              (take 3 chunks)))

                    :else
                    {:entities [] :triples []})

                  ;; 5. Store in graph
                  _ (graph/transact-extraction! knowledge-graph extraction :source-url target-url)

                  ;; 6. Score and enqueue new links
                  scored-links (mapv (fn [link]
                                       {:url (:url link)
                                        :efe-score (score-url link knowledge-goals knowledge-graph mode)})
                                     links)
                  crawler-with-links (crawler/enqueue-urls! crawler-after-fetch scored-links)

                  ;; 7. Build observation vector
                  obs-vector (build-observation-vector extraction knowledge-graph knowledge-goals text)

                  ;; Update stats
                  updated-forager (-> forager-state
                                      (assoc :crawler-state crawler-with-links)
                                      (assoc :knowledge-graph knowledge-graph)
                                      (update :step-count inc)
                                      (update-in [:stats :pages-fetched] inc)
                                      (update-in [:stats :entities-extracted] + (count (:entities extraction)))
                                      (update-in [:stats :triples-extracted] + (count (:triples extraction))))]
              [updated-forager obs-vector])))))))

;; --- Goal Management ---

(defn set-knowledge-goal
  "Sets the forager's knowledge goals."
  [forager-state goals]
  (assoc forager-state :knowledge-goals (if (string? goals) [goals] goals)))

;; --- Seeding ---

(defn seed-from-url
  "Seeds the forager's frontier with a starting URL."
  [forager-state url]
  (update forager-state :crawler-state
          crawler/enqueue-urls! [{:url url :efe-score 0.0}]))

(defn seed-from-wikipedia
  "Seeds the forager with a Wikipedia article URL for a given topic."
  [forager-state topic]
  (let [url (str "https://en.wikipedia.org/wiki/" (str/replace topic #"\s+" "_"))]
    (seed-from-url forager-state url)))

(defn seed-from-urls
  "Seeds the forager with multiple URLs."
  [forager-state urls]
  (update forager-state :crawler-state
          crawler/enqueue-urls!
          (mapv (fn [url] {:url url :efe-score 0.0}) urls)))

;; --- Multi-Step Foraging ---

(defn run-foraging
  "Runs N autonomous foraging steps. Returns the final forager state."
  [forager-state n-steps]
  (loop [state forager-state
         remaining n-steps]
    (if (zero? remaining)
      state
      (let [[updated-state obs] (forage-step state)]
        (recur updated-state (dec remaining))))))

;; --- Sleep & Consolidation ---

(defn- find-merge-candidates
  "Groups potential synonyms by simple similarity (shared words)."
  [names]
  (let [words (fn [n] (set (str/split (str/lower-case n) #"\s+")))
        groups (reduce (fn [acc name]
                         (let [n-words (words name)
                               match-idx (first (keep-indexed (fn [idx group]
                                                                (when (some #(seq (set/intersection n-words (words %))) group)
                                                                  idx))
                                                              acc))]
                           (if match-idx
                             (update acc match-idx conj name)
                             (conj acc [name]))))
                       []
                       names)]
    (filter #(> (count %) 1) groups)))

(defn sleep-consolidate
  "Sleep phase for knowledge consolidation:
   1. Detect duplicate/synonym entities (LLM-driven merging)
   2. Confidence decay for unconfirmed triples
   3. Find knowledge gaps
   4. Generate new search queries from gaps
   
   Returns [updated-state merges-performed]"
  [forager-state]
  (let [{:keys [knowledge-graph knowledge-goals llm-completion-fn]} forager-state

        ;; 1. Confidence Decay
        _ (graph/decay-confidence! knowledge-graph 0.05 0.3)

        ;; 2. Entity Merging
        all-entities (graph/query-all-entities knowledge-graph)
        entity-names (mapv first all-entities)
        merge-groups (find-merge-candidates entity-names)

        merges (when (and llm-completion-fn (seq merge-groups))
                 (reduce (fn [acc group]
                           (let [completion-fn (if (fn? llm-completion-fn)
                                                 llm-completion-fn
                                                 (fn [p] (llm/extract-knowledge llm-completion-fn p [])))
                                 merges (knowledge/canonicalize-entities completion-fn group entity-names)]
                             (doseq [{:keys [old-name canonical-name]} merges]
                               (graph/merge-entities! knowledge-graph old-name canonical-name))
                             (into acc merges)))
                         []
                         merge-groups))

        ;; 3. Find gaps
        gaps (graph/find-sparse-regions knowledge-graph 1)
        gap-names (mapv :entity-name gaps)

        ;; 4. Generate search queries from gaps (if LLM available)
        new-queries (when (and llm-completion-fn (seq gap-names))
                      (let [completion-fn (if (fn? llm-completion-fn)
                                            llm-completion-fn
                                            (fn [p] (llm/formulate-query llm-completion-fn gap-names knowledge-goals)))]
                        [(knowledge/formulate-search-query completion-fn
                                                           gap-names
                                                           knowledge-goals)]))

        ;; 5. Seed frontier with gap-driven queries (as Wikipedia articles for now)
        updated-state (if (seq new-queries)
                        (reduce (fn [s q]
                                  (seed-from-wikipedia s q))
                                forager-state
                                new-queries)
                        forager-state)]
    [(assoc updated-state :last-sleep-gaps gap-names) (or merges [])]))

;; --- Integration with Active Inference Core ---

(defn forager->observation
  "Converts forager state into an observation suitable for core/step."
  [forager-state]
  (let [kg (:knowledge-graph forager-state)
        embedding (graph/knowledge-embedding kg)]
    embedding))

(defn forager-stats
  "Returns a summary of the forager's current state."
  [forager-state]
  (let [{:keys [knowledge-graph crawler-state step-count stats knowledge-goals]} forager-state]
    (merge stats
           {:step-count step-count
            :knowledge-goals knowledge-goals
            :frontier-size (crawler/frontier-size crawler-state)
            :visited-count (crawler/visited-count crawler-state)
            :graph-stats (graph/graph-stats knowledge-graph)})))

(defn parse-action
  "Parses a string action from the AI core into a forager command.
   Example actions: 'VISIT:https://example.com', 'SEARCH:Active Inference'"
  [action-str]
  (cond
    (nil? action-str) {:type :wait}
    (not (string? action-str)) {:type :unknown :raw action-str}
    (str/starts-with? action-str "VISIT:") {:type :visit :url (subs action-str 6)}
    (str/starts-with? action-str "SEARCH:") {:type :search :query (subs action-str 7)}
    :else {:type :unknown :raw action-str}))

(defn execute-action
  "Dispatches a parsed action. Returns [updated-forager-state observation-vector]."
  [forager-state action]
  (case (:type action)
    :visit (forage-step forager-state :url (:url action))
    :search (let [search-url (crawler/search-wikipedia (:query action))]
              (forage-step forager-state :url search-url :search-engine :wikipedia))
    ;; default: pick from frontier
    (forage-step forager-state)))
