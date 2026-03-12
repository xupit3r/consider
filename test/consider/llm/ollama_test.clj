(ns consider.llm.ollama-test
  (:require [clojure.test :refer :all]
            [consider.llm.ollama :as ollama]
            [consider.llm :as llm]
            [hato.client :as http]
            [charred.api :as json]))

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

(deftest test-ollama-error-handling
  (testing "Proper handling of API errors"
    (let [llm-inst (ollama/make-ollama-llm "llama3")]
      (with-redefs [http/post (fn [_ _] {:status 500 :body "Internal Server Error"})]
        (is (thrown? Exception (llm/predict-candidates llm-inst [])))))))
