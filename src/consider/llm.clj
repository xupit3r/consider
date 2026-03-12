(ns consider.llm
  "Implementation of the LLM Abstraction Layer (System 1).
   Supports multiple providers:
   1. MockLLM (for deterministic testing)
   2. Ollama (for local inference, see consider.llm.ollama)
   3. DynamicLLM (generic wrapper for arbitrary completion functions)"
  (:require [clojure.spec.alpha :as s]
            [consider.specs.llm :as llm-spec]
            [clojure.string :as str]
            [charred.api :as json]))

;; --- Protocols ---

(defprotocol PolicyPredictor
  "Protocol for generating candidate reasoning steps and their priors."
  (predict-candidates [this context] "Returns a set of candidate-steps."))

(defprotocol ProcessScorer
  "Protocol for scoring a candidate step or path (Process Reward)."
  (score-step [this context candidate-action] "Returns a pragmatic and epistemic estimate."))

;; --- Prompt Templates ---

(defn- format-context
  [context]
  (str/join "\n" (map (fn [{:keys [role content]}] (str (str/upper-case (name role)) ": " content)) context)))

(defn prediction-prompt
  "Generates a prompt for System 1 policy prior generation."
  [context]
  (str "You are System 1, a fast policy predictor for an Active Inference agent.
Current trajectory:
" (format-context context) "

Generate 3-5 potential next reasoning steps. For each step, provide:
1. candidate-action: The text of the action.
2. prior-prob: Your confidence that this is the best next step (0.0 to 1.0).
3. pragmatic-estimate: Expected utility (0.0 to 1.0).
4. epistemic-estimate: Expected information gain (0.0 to 1.0).

Return ONLY a JSON array of objects with these keys."))

(defn scoring-prompt
  "Generates a prompt for scoring a specific candidate step."
  [context candidate-action]
  (str "You are a Process Reward Model. Evaluate the following candidate reasoning step:
Trajectory:
" (format-context context) "
Candidate Action: " candidate-action "

Provide:
1. pragmatic-estimate: Relevance to the final goal (0.0 to 1.0).
2. epistemic-estimate: Quality of reasoning/information gain (0.0 to 1.0).
3. confidence: Your certainty in this evaluation (0.0 to 1.0).

Return ONLY a JSON object with these keys."))

;; --- Mock Implementation for Testing ---

(defrecord MockLLM [responses]
  PolicyPredictor
  (predict-candidates [this context]
    (let [resp (get responses context)]
      (cond
        (fn? resp) (resp context)
        (vector? resp) resp
        :else [{:candidate-action "Standard Reasoning Step"
                :prior-prob 1.0
                :pragmatic-estimate 0.5
                :epistemic-estimate 0.5
                :confidence 1.0}])))

  ProcessScorer
  (score-step [this context candidate-action]
    (let [resp (get responses [context candidate-action])]
      (cond
        (fn? resp) (resp context candidate-action)
        (map? resp) resp
        :else {:pragmatic-estimate 0.5
               :epistemic-estimate 0.5
               :confidence 1.0}))))

(defn make-mock-llm
  "Creates a mock LLM with predefined responses."
  ([] (make-mock-llm {}))
  ([responses] (->MockLLM responses)))

;; --- Robust JSON Parsing & Sanitization ---

(defn- extract-json-from-markdown
  "Extracts JSON string from potential markdown code blocks."
  [s]
  (if-let [match (re-find #"(?s)```json\n?(.*?)\n?```" s)]
    (second match)
    (if-let [match (re-find #"(?s)```\n?(.*?)\n?```" s)]
      (second match)
      s)))

(defn- clamp [v min-v max-v]
  (max min-v (min max-v v)))

(defn- sanitize-math-components
  "Ensures all probability and estimate fields are clamped between 0 and 1."
  [m]
  (reduce-kv (fn [acc k v]
               (if (and (number? v) (re-find #"prob|estimate|confidence" (name k)))
                 (assoc acc k (clamp (double v) 0.0 1.0))
                 (assoc acc k v)))
             {}
             m))

(defn robust-parse-json
  "Attempts to parse JSON from LLM response, handling markdown and providing defaults."
  [raw-resp default-val]
  (let [cleaned (some-> raw-resp str/trim extract-json-from-markdown)
        parsed (try
                 (json/read-json cleaned :key-fn keyword)
                 (catch Exception _ nil))]
    (cond
      (vector? parsed) (mapv #(merge (if (vector? default-val) (first default-val) default-val)
                                     (sanitize-math-components %))
                             parsed)
      (map? parsed) (let [base (if (vector? default-val) (first default-val) default-val)
                          merged (merge base (sanitize-math-components parsed))]
                      (if (vector? default-val) [merged] merged))
      :else (if (vector? default-val) default-val [default-val]))))

(defrecord DynamicLLM [model-name api-key provider completion-fn]
  PolicyPredictor
  (predict-candidates [this context]
    (let [prompt (prediction-prompt context)
          raw-resp (completion-fn {:model model-name :prompt prompt})
          default [{:candidate-action "Error: Invalid response"
                    :prior-prob 0.0
                    :pragmatic-estimate 0.0
                    :epistemic-estimate 0.0
                    :confidence 0.0}]]
      (robust-parse-json raw-resp default)))

  ProcessScorer
  (score-step [this context candidate-action]
    (let [prompt (scoring-prompt context candidate-action)
          raw-resp (completion-fn {:model model-name :prompt prompt})
          default {:pragmatic-estimate 0.0
                   :epistemic-estimate 0.0
                   :confidence 0.0}]
      (robust-parse-json raw-resp default))))

(defn validate-candidate-step
  "Validates a candidate step against its specification."
  [step]
  (if (s/valid? ::llm-spec/candidate-step step)
    step
    (throw (ex-info "Invalid candidate step"
                    {:explain (s/explain-data ::llm-spec/candidate-step step)}))))
