(ns consider.core
  "Unified Reasoning Loop for the Consider Active Inference Agent."
  (:require [consider.world-model :as wm]
            [consider.inference :as inf]
            [consider.causal :as causal]
            [consider.executive :as exec]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]))

(defn- estimate-precision-matrix
  "Estimates the precision matrix from the current belief state.
   In a full implementation, this would use the history of internal states."
  [belief-state]
  (let [internal-states (:internal-states belief-state)
        d (count internal-states)
        theta (native/dge d d)]
    (n/scal! 0.0 theta)
    (dotimes [i d] (n/entry! theta i i 1.0))
    ;; Perturb slightly to avoid exact zero matrices if needed
    (dotimes [i d]
      (n/entry! theta i i (+ (n/entry theta i i) 1e-6)))
    (n/copy theta)))

(defn step
  "Performs a single Active Inference cycle: Perceive -> Infer -> Learn -> Decide -> Act."
  [agent-state sensory-data {:keys [inference-steps reasoning-iterations exploration-weight prune-threshold]}]
  (let [{:keys [belief-state orchestrator-state likelihood-fn vector-field-fn llm-system]} agent-state
        
        ;; 1. PERCEIVE & INFER: Update beliefs based on sensory data (Minimize VFE)
        updated-belief (inf/belief-update belief-state sensory-data likelihood-fn vector-field-fn inference-steps)
        
        ;; 2. LEARN: Discover causal structure from updated beliefs
        precision-matrix (estimate-precision-matrix updated-belief)
        causal-structure (causal/learn-structure precision-matrix)
        ;; Close the loop: Update world model's transitions with learned structure
        belief-with-learning (wm/update-transition-dynamics updated-belief (:sparse-S causal-structure))
        
        ;; 3. DECIDE: Perform MCTS reasoning to minimize Expected Free Energy (G)
        ;; Update orchestrator with the new belief trajectory
        initial-trajectory (conj [] {:role :user :content (str "Sensory Observation: " sensory-data)})
        updated-orchestrator (exec/add-tree orchestrator-state :current initial-trajectory)
        
        reasoned-orchestrator (exec/reason updated-orchestrator 
                                           llm-system ;; as Predictor
                                           llm-system ;; as Scorer
                                           :current 
                                           reasoning-iterations 
                                           exploration-weight 
                                           :prune-threshold prune-threshold)
        
        ;; 4. ACT: Extract the best policy
        best-policy (exec/extract-best-policy reasoned-orchestrator :current)
        next-action (first best-policy)]
    
    {:belief-state belief-with-learning
     :orchestrator-state reasoned-orchestrator
     :causal-structure causal-structure
     :next-action next-action
     :policy best-policy}))

(defn initialize-agent
  "Initializes the global agent state."
  [initial-beliefs preferences likelihood-fn vector-field-fn llm-system]
  {:belief-state (wm/make-belief-state initial-beliefs preferences)
   :orchestrator-state (exec/make-initial-orchestrator-state [])
   :likelihood-fn likelihood-fn
   :vector-field-fn vector-field-fn
   :llm-system llm-system})
