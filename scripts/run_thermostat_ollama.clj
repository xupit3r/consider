(ns run-thermostat-ollama
  (:require [consider.core :as core]
            [consider.world-model :as wm]
            [consider.llm.ollama :as ollama]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]))

(defn -main [& args]
  (let [model "llama3"
        target-temp 22.0
        initial-states {:room (wm/make-slot :room [18.0 0.0])}
        preferences [[target-temp 0.0]]
        likelihood-fn (fn [states]
                        (let [room (:room states)
                              pos (:position room)]
                          [(double (nth pos 0)) (double (nth pos 1))]))
        v-fn (fn [x t ctx] (native/dv (n/dim x)))
        
        ;; Initialize REAL Ollama LLM
        real-llm (ollama/make-ollama-llm model)
        
        agent (core/initialize-agent initial-states preferences likelihood-fn v-fn real-llm)]
    
    (println "--- Consider Agent: Smart Thermostat (Ollama) ---")
    (println "Model:" model)
    (println "Initial Temp: 18.0")
    (println "Target Temp: 22.0")
    (println "--------------------------------------------------")
    
    (loop [curr-agent agent
           i 1]
      (when (<= i 5)
        (let [belief (:belief-state curr-agent)
              room-state (get-in belief [:internal-states :room])
              current-temp (first (:position room-state))
              
              ;; Run the agent step
              _ (println "\n[Step" i "] Current Temp:" current-temp)
              res (core/step curr-agent [current-temp 0.0] 
                             {:inference-steps 5
                              :reasoning-iterations 10
                              :exploration-weight 1.0})
              
              next-action (:next-action res)]
          
          (println "LLM Suggested Action:" next-action)
          (println "EFE (G):" (:expected-free-energy (:belief-state res)))
          
          ;; Simulate simple environmental reaction
          (let [new-temp (if (= next-action "TURN_ON_HEATER")
                           (+ current-temp 1.0)
                           (- current-temp 0.2))]
            (recur (assoc curr-agent
                          :belief-state (:belief-state res)
                          :orchestrator-state (:orchestrator-state res)
                          :vector-field-fn (:vector-field-fn res))
                   (inc i))))))))

(when (= *file* (System/getProperty "babashka.file"))
  (apply -main *command-line-args*))
