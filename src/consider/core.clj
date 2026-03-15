(ns consider.core
  "Unified Reasoning Loop for the Consider Active Inference Agent."
  (:require [consider.world-model :as wm]
            [consider.inference :as inf]
            [consider.causal :as causal]
            [consider.executive :as exec]
            [consider.llm :as llm]
            [consider.models :as models]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.linalg :as la]
            [uncomplicate.neanderthal.real :as r]))

(defn- estimate-precision-matrix
  "Estimates the precision matrix from the history of internal states.
   Uses shrinkage (L2 regularization) to ensure invertibility.
   Flatten all dimensions of all slots for the precision matrix."
  [belief-state]
  (let [history (:history belief-state)
        internal-states (:internal-states belief-state)
        slot-ids (sort (keys internal-states))

        ;; Calculate total dimensions across all slots
        all-dims (map (fn [id] (count (:position (get internal-states id)))) slot-ids)
        total-d (reduce + all-dims)
        n-samples (count history)
        ;; Increased shrinkage for better stability in stress scenarios
        lambda 1e-2]

    (if (< n-samples 10)
      ;; Fallback to identity with slight perturbation
      (let [theta (native/dge total-d total-d)]
        (n/scal! 0.0 theta)
        (dotimes [i total-d] (n/entry! theta i i (+ 1.0 lambda)))
        theta)
      (let [X (native/dge n-samples total-d)]
        ;; 1. Populate data matrix X by flattening all dimensions
        (dotimes [i n-samples]
          (let [sample (nth history i)
                states (:internal-states sample)]
            (loop [ids slot-ids
                   col-offset 0]
              (when-let [id (first ids)]
                (let [pos (or (:position (get states id))
                              (vec (repeat (count (:position (get internal-states id))) 0.0)))
                      dim (count pos)]
                  (dotimes [k dim]
                    (n/entry! X i (+ col-offset k) (double (nth pos k))))
                  (recur (rest ids) (+ col-offset dim)))))))

        ;; 2. Center X
        (let [means (native/dv total-d)]
          (n/scal! 0.0 means)
          (dotimes [j total-d]
            (let [col (n/submatrix X 0 j n-samples 1)]
              (n/entry! means j (/ (n/asum col) n-samples))))
          (dotimes [i n-samples]
            (dotimes [j total-d]
              (n/entry! X i j (- (n/entry X i j) (n/entry means j))))))

        ;; 3. Compute sample covariance C = (X^T * X) / (n-1)
        (let [C (n/mm (n/trans X) X)]
          (n/scal! (/ 1.0 (max 1.0 (dec n-samples))) C)

          ;; 4. Shrinkage: C = C + lambda * I
          (dotimes [i total-d]
            (n/entry! C i i (+ (n/entry C i i) lambda)))

          ;; 5. Invert covariance to get precision matrix Theta
          (let [theta (try
                        (let [I (native/dge total-d total-d)]
                          (n/scal! 0.0 I)
                          (dotimes [i total-d] (n/entry! I i i 1.0))
                          (la/sv C I))
                        (catch Exception _
                          (let [t (native/dge total-d total-d)]
                            (n/scal! 0.0 t)
                            (dotimes [i total-d] (n/entry! t i i (+ 1.0 lambda)))
                            t)))]

            ;; SANITY CHECK: If result has NaN, fallback to Identity
            (if (Double/isNaN (r/nrm2 theta))
              (let [t (native/dge total-d total-d)]
                (n/scal! 0.0 t)
                (dotimes [i total-d] (n/entry! t i i (+ 1.0 lambda)))
                t)
              (do
                ;; Clip values in the precision matrix to prevent explosive feedback
                (dotimes [i total-d]
                  (dotimes [j total-d]
                    (let [v (n/entry theta i j)]
                      (n/entry! theta i j (max -100.0 (min 100.0 v))))))
                theta))))))))

(defn step
  "Performs a single Active Inference cycle: Perceive -> Infer -> Learn -> Decide -> Act."
  [agent-state sensory-data {:keys [inference-steps reasoning-iterations exploration-weight prune-threshold merge-threshold]}]
  (let [{:keys [belief-state orchestrator-state likelihood-fn vector-field-fn llm-system]} agent-state

        ;; 1. GROWTH CHECK: Detect novelty
        predicted-obs (wm/predict-observation belief-state)
        novel-slots (wm/identify-novel-entities belief-state sensory-data predicted-obs)

        belief-after-growth (if (empty? novel-slots)
                              belief-state
                              (wm/grow-slots belief-state novel-slots))

        ;; 2. MERGE CHECK: Detect redundancy (Consolidation)
        redundant-groups (wm/identify-redundant-slots belief-after-growth (or merge-threshold 0.01))

        belief-after-consolidation (if (empty? redundant-groups)
                                     belief-after-growth
                                     (reduce (fn [bs {:keys [target sources]}]
                                               (wm/merge-slots bs target sources))
                                             belief-after-growth
                                             redundant-groups))

        ;; 3. STRUCTURAL ADAPTATION: Resize neural network if state space changed
        vector-field-after-growth
        (if (and (not (fn? vector-field-fn))
                 (or (seq novel-slots) (seq redundant-groups)))
          (let [all-slots (:internal-states belief-after-consolidation)
                total-state-dim (reduce + (map #(count (:position %)) (vals all-slots)))
                total-obs-dim (count (if (seqable? sensory-data) sensory-data [sensory-data]))]
            (models/resize-network vector-field-fn total-state-dim total-obs-dim))
          vector-field-fn)

        ;; 4. PERCEIVE & INFER: Update beliefs based on sensory data (Minimize VFE)
        sensory-v (if (vector? sensory-data) sensory-data [sensory-data])
        updated-belief (inf/belief-update belief-after-consolidation sensory-v likelihood-fn vector-field-after-growth inference-steps)

        ;; 5. LEARN: Discover causal structure and hierarchical abstractions
        precision-matrix (estimate-precision-matrix updated-belief)
        causal-structure (causal/learn-structure precision-matrix)

        ;; Level 2: Group slots into concepts
        concept-modules (causal/learn-hierarchy causal-structure (keys (:internal-states updated-belief)))

        belief-with-learning (-> updated-belief
                                 (wm/update-transition-dynamics (:sparse-S causal-structure))
                                 (assoc :hierarchy {:conceptual-states (into {} (map (fn [c] [(:concept-id c) c]) concept-modules))}))

        ;; 6. DECIDE: Perform MCTS reasoning to minimize Expected Free Energy (G)
        initial-trajectory (conj [] {:role :user :content (str "Sensory Observation: " sensory-v)})
        updated-orchestrator (-> orchestrator-state
                                 (exec/add-tree :current initial-trajectory)
                                 (with-meta {:belief-state (if (:likelihood-mapping belief-with-learning)
                                                             belief-with-learning
                                                             (wm/with-generative-model belief-with-learning likelihood-fn (fn [s a] s)))
                                             :precision-matrix (:precision-matrix causal-structure)
                                             :slot-ids (sort (keys (:internal-states belief-with-learning)))
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

        ;; 7. ACT: Extract the best policy
        best-policy (exec/extract-best-policy reasoned-orchestrator :current)
        next-action (first best-policy)

        ;; 8. SLEEP (Amortization): Update the recognition model
        trained-vector-field (inf/train-recognition-model vector-field-after-growth belief-with-learning 10)]

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

(defn initialize-foraging-agent
  "Initializes an agent configured for web foraging.
   Sets up knowledge-specific likelihood/transition functions and
   a 6-dimensional observation space for knowledge metrics.

   Options:
     :knowledge-goals - vector of topic strings
     :llm-system - LLM implementing PolicyPredictor + ProcessScorer
     :vector-field-fn - flow field for inference (or nil for default)"
  [opts]
  (let [goals (or (:knowledge-goals opts) [])
        llm-system (:llm-system opts)
        ;; Knowledge domain starts with one slot per goal (or one default)
        initial-slots (if (seq goals)
                        (into {} (map-indexed
                                  (fn [i goal]
                                    (let [id (keyword (str "domain-" i))]
                                      [id (wm/make-knowledge-slot id
                                                                  [0.0 0.0 0.0 0.0 0.5 0.5]
                                                                  [1.0 1.0 1.0 1.0 1.0 1.0])]))
                                  goals))
                        {:domain-0 (wm/make-knowledge-slot :domain-0
                                                           [0.0 0.0 0.0 0.0 0.5 0.5]
                                                           [1.0 1.0 1.0 1.0 1.0 1.0])})
        ;; Preferences: goal-like observations (high topic similarity, high quality)
        preferences (if (seq goals)
                      [[0.0 5.0 0.0 0.0 1.0 1.0]]
                      [])
        likelihood-fn (wm/knowledge-likelihood-fn nil)
        transition-fn (wm/knowledge-transition-fn)
        vector-field-fn (or (:vector-field-fn opts)
                            (fn [x t context] (native/dv (n/dim x))))]
    (initialize-agent initial-slots preferences likelihood-fn vector-field-fn llm-system)))
