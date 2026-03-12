(ns consider.llm.ollama-test
  (:require [clojure.test :refer :all]
            [consider.llm.ollama :as ollama]
            [consider.llm :as llm]
            [consider.core :as core]
            [consider.world-model :as wm]
            [hato.client :as http]
            [charred.api :as json]
            [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]))

(deftest test-ollama-policy-prediction
  (testing "Successful policy prediction through Ollama mock"
    (let [mock-response (json/write-json-str
                         {:response (json/write-json-str
                                     [{:candidate-action "Action 1" :prior-prob 0.9 :pragmatic-estimate 0.8 :epistemic-estimate 0.1}])})
          llm-inst (ollama/make-ollama-llm "llama3")]
      
      (with-redefs [http/post (fn [url opts]
                                (is (clojure.string/includes? url "/api/generate"))
                                (let [body (json/read-json (:body opts) :key-fn keyword)]
                                  (is (= "llama3" (:model body)))
                                  (is (some? (:prompt body)))
                                  (is (= "json" (:format body))))
                                {:status 200 :body mock-response})]
        
        (let [candidates (llm/predict-candidates llm-inst [{:role :user :content "Hello"}])]
          (is (= 1 (count candidates)))
          (is (= "Action 1" (:candidate-action (first candidates))))
          (is (= 0.9 (:prior-prob (first candidates)))))))))

(deftest test-ollama-scoring
  (testing "Successful process scoring through Ollama mock"
    (let [mock-response (json/write-json-str
                         {:response (json/write-json-str
                                     {:pragmatic-estimate 0.7 :epistemic-estimate 0.3 :confidence 1.0})})
          llm-inst (ollama/make-ollama-llm "llama3")]
      
      (with-redefs [http/post (fn [url opts]
                                {:status 200 :body mock-response})]
        
        (let [score (llm/score-step llm-inst [{:role :user :content "Context"}] "Action")]
          (is (= 0.7 (:pragmatic-estimate score)))
          (is (= 0.3 (:epistemic-estimate score)))
          (is (= 1.0 (:confidence score))))))))

(deftest test-ollama-robust-parsing
  (testing "Ollama LLM handles markdown-wrapped JSON and out-of-bounds values"
    (let [markdown-resp (str "Certainly! Here is the JSON:\n"
                             "```json\n"
                             "{\"candidate-action\": \"Markdown Action\", \"prior-prob\": 1.5, \"pragmatic-estimate\": -0.5, \"epistemic-estimate\": 0.5}\n"
                             "```")
          mock-response (json/write-json-str {:response markdown-resp})
          llm-inst (ollama/make-ollama-llm "llama3")]
      
      (with-redefs [http/post (fn [_ _] {:status 200 :body mock-response})]
        (let [candidates (llm/predict-candidates llm-inst [])]
          (is (= 1 (count candidates)))
          (is (= "Markdown Action" (:candidate-action (first candidates))))
          (is (= 1.0 (:prior-prob (first candidates))) "Should clamp 1.5 to 1.0")
          (is (= 0.0 (:pragmatic-estimate (first candidates))) "Should clamp -0.5 to 0.0"))))))

(deftest test-ollama-error-handling
  (testing "Proper handling of API errors"
    (let [llm-inst (ollama/make-ollama-llm "llama3")]
      (with-redefs [http/post (fn [_ _] {:status 500 :body "Internal Server Error"})]
        (is (thrown? Exception (llm/predict-candidates llm-inst [])))))))

(deftest test-ollama-integration-with-core
  (testing "Full core/step cycle with Ollama LLM"
    (let [initial-states {:me (wm/make-slot :me [0.0])}
          preferences [[1.0]]
          likelihood-fn (fn [states] [0.0])
          v-fn (fn [x t ctx] (native/dv (n/dim x)))
          
          ;; Mock Ollama response for both prediction and scoring
          prediction-resp (json/write-json-str 
                           {:response (json/write-json-str
                                       [{:candidate-action "MOVE_FORWARD" :prior-prob 0.9 
                                         :pragmatic-estimate 0.8 :epistemic-estimate 0.1}])})
          scoring-resp (json/write-json-str 
                        {:response (json/write-json-str
                                    {:pragmatic-estimate 0.9 :epistemic-estimate 0.1 :confidence 1.0})})
          
          llm-inst (ollama/make-ollama-llm "llama3")
          agent (core/initialize-agent initial-states preferences likelihood-fn v-fn llm-inst)]
      
      (with-redefs [http/post (fn [url _]
                                (cond
                                  (clojure.string/includes? url "/api/generate")
                                  {:status 200 :body (if (>= (count (re-find #"Evaluate" url)) 0) ;; crude way to distinguish prompts in mock
                                                       ;; scoring prompt contains "Evaluate"
                                                       scoring-resp 
                                                       prediction-resp)}
                                  :else {:status 404}))]
        
        ;; Note: The above redefs needs to be smarter about which prompt is being served.
        ;; Let's fix the mock logic:
        (with-redefs [http/post (fn [_ opts]
                                  (let [prompt (get (json/read-json (:body opts) :key-fn keyword) :prompt)]
                                    {:status 200 
                                     :body (if (clojure.string/includes? prompt "Evaluate")
                                             scoring-resp
                                             prediction-resp)}))]
          
          (let [res (core/step agent [0.0] {:inference-steps 1
                                           :reasoning-iterations 2
                                           :exploration-weight 1.0})]
            (is (= "MOVE_FORWARD" (:next-action res)))
            (is (some? (:causal-structure res)))))))))
