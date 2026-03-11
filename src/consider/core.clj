(ns consider.core
  "Unified Reasoning Loop for the Consider Active Inference Agent."
  (:require [consider.world-model :as wm]
            [consider.inference :as inf]
            [consider.causal :as causal]
            [consider.executive :as exec]
            [consider.llm :as llm]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.linalg :as la]))

(defn- estimate-precision-matrix
  "Estimates the precision matrix from the history of internal states.
   Uses shrinkage (L2 regularization) to ensure invertibility."
  [belief-state]
  (let [history (:history belief-state)
        internal-states (:internal-states belief-state)
        slot-ids (sort (keys internal-states))
        d (count slot-ids)
        n-samples (count history)
        lambda 1e-4]
    (if (< n-samples 2)
      ;; Fallback to identity with slight perturbation
      (let [theta (native/dge d d)]
        (n/scal! 0.0 theta)
        (dotimes [i d] (n/entry! theta i i (+ 1.0 lambda)))
        theta)
      (let [X (native/dge n-samples d)]
        ;; 1. Populate data matrix X
        (dotimes [i n-samples]
          (let [sample (nth history i)
                states (:internal-states sample)]
            (dotimes [j d]
              (let [id (nth slot-ids j)
                    ;; Use the first dimension of position for now
                    val (first (:position (get states id)))]
                (n/entry! X i j (or val 0.0))))))
        
        ;; 2. Center X
        (let [means (native/dv d)]
          (n/scal! 0.0 means)
          (dotimes [j d]
            (let [col (n/submatrix X 0 j n-samples 1)]
              (n/entry! means j (/ (n/asum col) n-samples))))
          (dotimes [i n-samples]
            (dotimes [j d]
              (n/entry! X i j (- (n/entry X i j) (n/entry means j))))))
        
        ;; 3. Compute sample covariance C = (X^T * X) / (n-1)
        (let [C (n/mm (n/trans X) X)]
          (n/scal! (/ 1.0 (dec n-samples)) C)
          
          ;; 4. Shrinkage: C = C + lambda * I
          (dotimes [i d]
            (n/entry! C i i (+ (n/entry C i i) lambda)))
          
          ;; 5. Invert covariance to get precision matrix Theta
          ;; We solve CX = I to get X = C^-1
          (let [I (native/dge d d)]
            (n/scal! 0.0 I)
            (dotimes [i d] (n/entry! I i i 1.0))
            (la/sv C I)))))))

(defn step
  "Performs a single Active Inference cycle: Perceive -> Infer -> Learn -> Decide -> Act."
  [agent-state sensory-data {:keys [inference-steps reasoning-iterations exploration-weight prune-threshold]}]
  (let [{:keys [belief-state orchestrator-state likelihood-fn vector-field-fn llm-system]} agent-state
        
        ;; 1. PERCEIVE & INFER: Update beliefs based on sensory data (Minimize VFE)
        updated-belief (inf/belief-update belief-state sensory-data likelihood-fn vector-field-fn inference-steps)
        
        ;; 1b. GROW: Check for novel entities if VFE/error is high
        predicted-obs (wm/predict-observation updated-belief)
        novel-slots (wm/identify-novel-entities updated-belief sensory-data predicted-obs)
        belief-after-growth (if (empty? novel-slots)
                              updated-belief
                              (wm/grow-slots updated-belief novel-slots))
        
        ;; 2. LEARN: Discover causal structure from updated beliefs
        precision-matrix (estimate-precision-matrix belief-after-growth)
        causal-structure (causal/learn-structure precision-matrix)
        ;; Close the loop: Update world model's transitions with learned structure
        belief-with-learning (wm/update-transition-dynamics belief-after-growth (:sparse-S causal-structure))
        
        ;; 3. DECIDE: Perform MCTS reasoning to minimize Expected Free Energy (G)
        ;; Update orchestrator with the new belief trajectory and causal epistemic guidance
        initial-trajectory (conj [] {:role :user :content (str "Sensory Observation: " sensory-data)})
        updated-orchestrator (-> orchestrator-state
                                 (exec/add-tree :current initial-trajectory)
                                 (with-meta {:belief-state belief-with-learning
                                             :causal-ambiguity-fn 
                                             (fn [trajectory]
                                               (let [acyclicity (:acyclicity causal-structure)
                                                     ambiguity-score (Math/exp (or acyclicity 0.0))]
                                                 (/ 1.0 (max 1e-6 ambiguity-score))))}))
        
        reasoned-orchestrator (exec/reason updated-orchestrator 
                                           llm-system ;; as Predictor
                                           llm-system ;; as Scorer
                                           :current 
                                           reasoning-iterations 
                                           exploration-weight 
                                           :prune-threshold prune-threshold
                                           :likelihood-fn (:likelihood-mapping belief-with-learning)
                                           :transition-fn (:transition-dynamics belief-with-learning))
        
        ;; 4. ACT: Extract the best policy
        best-policy (exec/extract-best-policy reasoned-orchestrator :current)
        next-action (first best-policy)
        
        ;; 5. SLEEP (Amortization): Update the recognition model
        trained-vector-field (inf/train-recognition-model vector-field-fn belief-with-learning 10)]
    
    {:belief-state belief-with-learning
     :orchestrator-state reasoned-orchestrator
     :vector-field-fn trained-vector-field
     :causal-structure causal-structure
     :next-action next-action
     :policy best-policy}))

(defn initialize-agent
  "Initializes the global agent state."
  [initial-beliefs preferences likelihood-fn vector-field-fn llm-system]
  {:belief-state (-> (wm/make-belief-state initial-beliefs preferences)
                     (wm/with-generative-model likelihood-fn (fn [s a] s)))
   :orchestrator-state (exec/make-initial-orchestrator-state [])
   :likelihood-fn likelihood-fn
   :vector-field-fn vector-field-fn
   :llm-system llm-system})
