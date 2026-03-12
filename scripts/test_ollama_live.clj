(ns test-ollama-live
  (:require [consider.llm.ollama :as ollama]
            [consider.llm :as llm]
            [clojure.pprint :refer [pprint]]))

(defn run-live-test []
  (let [model "qwen3:8b" ;; Use one of the available models
        llm-inst (ollama/make-ollama-llm model)
        context [{:role :user :content "I am in a room with a closed door and a key on the table."}]]
    
    (println "--- Testing Policy Prediction (Live) ---")
    (try
      (let [candidates (llm/predict-candidates llm-inst context)]
        (pprint candidates))
      (catch Exception e
        (println "Prediction Error:" (.getMessage e))))

    (println "\n--- Testing Process Scoring (Live) ---")
    (try
      (let [score (llm/score-step llm-inst context "Pick up the key")]
        (pprint score))
      (catch Exception e
        (println "Scoring Error:" (.getMessage e))))))

(run-live-test)
(shutdown-agents)
