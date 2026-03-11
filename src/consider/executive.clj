(ns consider.executive
  "Implementation of the Executive Orchestrator (MCTS & Forest-of-Thought)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.executive :as exec-spec]
            [consider.llm :as llm]))

(defn make-node
  "Creates a new MCTS node."
  [{:keys [node-id parent-id state action prior-prob risk ambiguity]}]
  (let [r (or risk 0.5)
        a (or ambiguity 0.5)]
    {:node-id node-id
     :parent-id parent-id
     :state state
     :action action
     ;; G = Risk + Ambiguity. We want to MINIMIZE this.
     :value (+ r a)
     :risk r
     :ambiguity a
     :visits 0
     :prior-prob prior-prob}))

(defn make-initial-orchestrator-state
  "Constructs the initial state for the orchestrator."
  [initial-state]
  (let [root-node (make-node {:node-id "root"
                              :parent-id nil
                              :state initial-state
                              :action "ROOT"
                              :prior-prob 1.0})]
    {:forest {:main {"root" root-node}}
     :active-branches #{"root"}
     :max-depth 10
     :max-compute-tokens 1000}))

(defn expand-node
  "Expands a node using candidate steps from the LLM."
  [orchestrator-state tree-id node-id candidates]
  (let [tree (get-in orchestrator-state [:forest tree-id])
        parent-node (get tree node-id)
        new-nodes (map-indexed
                   (fn [idx candidate]
                     (let [new-id (str node-id "-" idx)
                           ;; Convert pragmatic utility to risk: risk = 1.0 - utility
                           risk (- 1.0 (or (:pragmatic-estimate candidate) 0.5))
                           ambiguity (or (:epistemic-estimate candidate) 0.5)]
                       (make-node {:node-id new-id
                                   :parent-id node-id
                                   :state (conj (:state parent-node)
                                                {:role :assistant
                                                 :content (:candidate-action candidate)})
                                   :action (:candidate-action candidate)
                                   :prior-prob (:prior-prob candidate)
                                   :risk risk
                                   :ambiguity ambiguity})))
                   candidates)]
    (update-in orchestrator-state [:forest tree-id]
               (fn [t] (reduce (fn [acc node] (assoc acc (:node-id node) node)) t new-nodes)))))

(defn select-best-node
  "Selects the next node to expand using UCT modified for EFE (minimizing value)."
  [tree node-id exploration-weight]
  (let [parent-node (get tree node-id)
        children (filter #(= (:parent-id %) node-id) (vals tree))]
    (if (empty? children)
      node-id
      (let [parent-visits (:visits parent-node)
            uct-fn (fn [node]
                     (let [v (:value node) ;; In Active Inference, we want to MINIMIZE G
                           n (:visits node)
                           p (:prior-prob node)]
                       (if (zero? n)
                         Double/NEGATIVE_INFINITY ;; Prioritize unvisited nodes
                         (- v (* exploration-weight p (Math/sqrt (/ (Math/log parent-visits) n)))))))]
        (-> (apply min-key uct-fn children)
            :node-id)))))

(defn update-node-value
  "Backpropagates value updates up the tree."
  [tree node-id new-value]
  (loop [curr-tree tree
         curr-id node-id]
    (if-not curr-id
      curr-tree
      (let [node (get curr-tree curr-id)
            old-visits (:visits node)
            new-visits (inc old-visits)
            updated-node (-> node
                             (assoc :visits new-visits)
                             (update :value (fn [old-v] (/ (+ (* old-v old-visits) new-value)
                                                           new-visits))))]
        (recur (assoc curr-tree curr-id updated-node)
               (:parent-id node))))))

(defn prune-branches
  "Prunes branches from the tree that have an Expected Free Energy (value) above the threshold."
  [orchestrator-state tree-id threshold]
  (let [tree (get-in orchestrator-state [:forest tree-id])
        ;; Find all nodes with value > threshold (we want to MINIMIZE value/EFE)
        nodes-to-prune (filter (fn [[id node]] (> (:value node) threshold)) tree)
        ids-to-prune (set (map first nodes-to-prune))

        ;; Function to find all descendants
        get-descendants (fn [t root-id]
                          (loop [queue [root-id]
                                 descendants #{}]
                            (if (empty? queue)
                              descendants
                              (let [curr (first queue)
                                    children (map :node-id (filter #(= (:parent-id %) curr) (vals t)))]
                                (recur (into (vec (rest queue)) children)
                                       (into descendants children))))))

        all-pruned-ids (reduce (fn [acc id] (into acc (get-descendants tree id))) ids-to-prune ids-to-prune)
        new-tree (apply dissoc tree (into all-pruned-ids ids-to-prune))]
    (assoc-in orchestrator-state [:forest tree-id] new-tree)))

(defn extract-best-policy
  "Extracts the sequence of actions for the branch with the lowest Expected Free Energy."
  [orchestrator-state tree-id]
  (let [tree (get-in orchestrator-state [:forest tree-id])
        ;; Find leaf nodes (nodes with no children)
        all-ids (set (keys tree))
        parent-ids (set (map :parent-id (vals tree)))
        leaf-nodes (filter #(not (contains? parent-ids (:node-id %))) (vals tree))

        best-leaf (if (empty? leaf-nodes)
                    (get tree "root")
                    (apply min-key :value leaf-nodes))]
    (loop [curr-node best-leaf
           actions []]
      (if (= "root" (:node-id curr-node))
        (reverse actions)
        (recur (get tree (:parent-id curr-node))
               (conj actions (:action curr-node)))))))

(defn add-tree
  "Adds a new reasoning tree to the forest."
  [orchestrator-state tree-id initial-state]
  (let [root-node (make-node {:node-id "root"
                              :parent-id nil
                              :state initial-state
                              :action "ROOT"
                              :prior-prob 1.0})]
    (assoc-in orchestrator-state [:forest tree-id] {"root" root-node})))

(defn- parse-interventional-action
  "Attempts to parse a string action into a structured intervention map.
   Format: 'DO(target-id, [val1, val2])'"
  [action-str]
  (try
    (if-let [match (re-find #"DO\((:[^,]+),\s*\[([^\]]+)\]\)" action-str)]
      (let [target (keyword (subs (second match) 1)) ;; Remove leading :
            values (mapv #(Double/parseDouble %) (clojure.string/split (last match) #",\s*"))]
        {:type :do :target target :value values})
      action-str)
    (catch Exception _ action-str)))

(defn reason
  "Performs a specified number of reasoning iterations using MCTS with sparse pruning.
   Uses the generative model (if provided) to refine the Expected Free Energy (G) calculation.
   Supports interventional actions (do-calculus)."
  [orchestrator-state predictor scorer tree-id iterations exploration-weight
   & {:keys [prune-threshold likelihood-fn transition-fn]}]
  (loop [state orchestrator-state
         i iterations]
    (if (zero? i)
      state
      (let [tree (get-in state [:forest tree-id])
            ;; 1. Selection
            leaf-id (select-best-node tree "root" exploration-weight)
            leaf (get tree leaf-id)
            ;; 2. Expansion (via LLM predictor)
            candidates (llm/predict-candidates predictor (:state leaf))
            state-after-expansion (expand-node state tree-id leaf-id candidates)
            ;; 3. Evaluation (Refined Expected Free Energy)
            avg-heuristic-value (let [avg-pragmatic (/ (reduce + (map :pragmatic-estimate candidates)) (max 1 (count candidates)))
                                      avg-epistemic (/ (reduce + (map :epistemic-estimate candidates)) (max 1 (count candidates)))
                                      ;; G = Risk - InformationGain. Risk = 1.0 - Utility
                                      llm-g (- (- 1.0 avg-pragmatic) avg-epistemic)

                                      ;; Model's rigorous G (if world model functions are available)
                                      model-g (if (and likelihood-fn transition-fn)
                                                (let [meta-bs (:belief-state (meta orchestrator-state))
                                                      action-str (or (:action leaf) "STAY")
                                                      ;; NEW: Support interventional parsing
                                                      action (parse-interventional-action action-str)

                                                      pred-states (transition-fn (:internal-states meta-bs) action)
                                                      pred-bs (assoc meta-bs :internal-states pred-states)
                                                      efe-res (try
                                                                (let [inf-ns (find-ns 'consider.inference)
                                                                      efe-fn (ns-resolve inf-ns 'expected-free-energy)]
                                                                  (efe-fn pred-bs likelihood-fn))
                                                                (catch Exception _ nil))]
                                                  (if efe-res
                                                    (- (:g efe-res) avg-epistemic)
                                                    llm-g))
                                                llm-g)

                                      causal-val (if-let [amb-fn (get (meta orchestrator-state) :causal-ambiguity-fn)]
                                                   (amb-fn (:state leaf))
                                                   0.0)]
                                  ;; Final G estimate: Model G adjusted by causal uncertainty reduction
                                  (- model-g causal-val))
            ;; 4. Backpropagation
            current-tree (get-in state-after-expansion [:forest tree-id])
            updated-tree (update-node-value current-tree leaf-id avg-heuristic-value)
            state-with-updated-tree (assoc-in state-after-expansion [:forest tree-id] updated-tree)

            ;; 5. Sparse Activation (Optional Pruning)
            final-state (if prune-threshold
                          (prune-branches state-with-updated-tree tree-id prune-threshold)
                          state-with-updated-tree)]
        (recur final-state (dec i))))))
