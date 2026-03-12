(ns consider.llm.ollama
  "Ollama-specific implementation of the LLM protocols."
  (:require [consider.llm :as llm]
            [hato.client :as http]
            [charred.api :as json]
            [promesa.core :as p]))

(defn- generate-completion
  "Calls the Ollama API to generate a completion.
   Ollama should be running on the provided base-url."
  [base-url model prompt]
  (let [url (str base-url "/api/generate")
        payload {:model model
                 :prompt prompt
                 :stream false
                 :format "json"}
        resp (http/post url {:body (json/write-json-str payload)
                             :content-type :json})]
    (if (= 200 (:status resp))
      (let [body (json/read-json (:body resp) :key-fn keyword)]
        (:response body))
      (throw (ex-info "Ollama API Error" {:status (:status resp) :body (:body resp)})))))

(defrecord OllamaLLM [model-name base-url]
  llm/PolicyPredictor
  (predict-candidates [this context]
    (let [prompt (llm/prediction-prompt context)
          raw-resp (generate-completion base-url model-name prompt)
          default [{:candidate-action "Error: Invalid response"
                    :prior-prob 0.0
                    :pragmatic-estimate 0.0
                    :epistemic-estimate 0.0
                    :confidence 0.0}]]
      (llm/robust-parse-json raw-resp default)))

  llm/ProcessScorer
  (score-step [this context candidate-action]
    (let [prompt (llm/scoring-prompt context candidate-action)
          raw-resp (generate-completion base-url model-name prompt)
          default {:pragmatic-estimate 0.0
                   :epistemic-estimate 0.0
                   :confidence 0.0}]
      (llm/robust-parse-json raw-resp default))))

(extend-type OllamaLLM
  llm/KnowledgeExtractor
  (extract-knowledge [this text entities]
    (let [completion-fn (fn [prompt] (generate-completion (:base-url this) (:model-name this) prompt))]
      (require '[consider.web.knowledge :as wk])
      ((resolve 'consider.web.knowledge/text-to-triples) completion-fn text entities)))
  (formulate-query [this gaps goals]
    (let [completion-fn (fn [prompt] (generate-completion (:base-url this) (:model-name this) prompt))]
      (require '[consider.web.knowledge :as wk])
      ((resolve 'consider.web.knowledge/formulate-search-query) completion-fn gaps goals))))

(defn make-ollama-completion-fn
  "Creates a simple completion function suitable for knowledge extraction.
   Returns a function: prompt-string -> response-string."
  ([model-name] (make-ollama-completion-fn model-name "http://localhost:11434"))
  ([model-name base-url]
   (fn [prompt] (generate-completion base-url model-name prompt))))

(defn make-ollama-llm
  "Creates an OllamaLLM instance.
   Defaults to localhost:11434 if no base-url is provided."
  ([model-name] (make-ollama-llm model-name "http://localhost:11434"))
  ([model-name base-url]
   (->OllamaLLM model-name base-url)))
