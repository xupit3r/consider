(ns consider.executive
  "Implementation of the Executive Orchestrator (MCTS & Forest-of-Thought)."
  (:require [clojure.spec.alpha :as s]
            [consider.specs.executive :as exec-spec]
            [consider.llm :as llm]))

(defn make-node
  "Creates a new MCTS node."
  [{:keys [node-id parent-id state action prior-prob risk ambiguity]}]
  {:node-id node-id
   :parent-id parent-id
   :state state
   :action action
   :value (+ (or risk 0.5) (or ambiguity 0.5)) ;; Initial value
   :risk (or risk 0.5)
   :ambiguity (or ambiguity 0.5)
   :visits 0
   :prior-prob prior-prob})

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
                     (let [new-id (str node-id "-" idx)]
                       (make-node {:node-id new-id
                                   :parent-id node-id
                                   :state (conj (:state parent-node) 
                                                {:role :assistant 
                                                 :content (:candidate-action candidate)})
                                   :action (:candidate-action candidate)
                                   :prior-prob (:prior-prob candidate)
                                   :risk (:pragmatic-estimate candidate)
                                   :ambiguity (:epistemic-estimate candidate)})))
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

(defn reason
  "Performs a specified number of reasoning iterations using MCTS."
  [orchestrator-state predictor scorer tree-id iterations exploration-weight]
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
            ;; 3. Evaluation (via LLM scorer - use average of candidates for simplicity)
            avg-heuristic-value (/ (reduce + (map (fn [c] (+ (:pragmatic-estimate c) (:epistemic-estimate c))) candidates))
                                   (max 1 (count candidates)))
            ;; 4. Backpropagation
            current-tree (get-in state-after-expansion [:forest tree-id])
            updated-tree (update-node-value current-tree leaf-id avg-heuristic-value)]
        (recur (assoc-in state-after-expansion [:forest tree-id] updated-tree)
               (dec i))))))
