(ns consider.web.crawler-test
  (:require [clojure.test :refer :all]
            [consider.web.crawler :as crawler]))

(deftest test-url-utilities
  (testing "normalize-url removes fragments"
    (is (= "https://example.com/page"
           (crawler/normalize-url "https://example.com/page#section"))))

  (testing "normalize-url handles basic URLs"
    (is (string? (crawler/normalize-url "https://example.com/path?q=1"))))

  (testing "extract-domain works"
    (is (= "example.com" (crawler/extract-domain "https://example.com/page")))
    (is (= "en.wikipedia.org" (crawler/extract-domain "https://en.wikipedia.org/wiki/Test"))))

  (testing "extract-domain returns nil for invalid URLs"
    (is (nil? (crawler/extract-domain "not-a-url"))))

  (testing "resolve-url resolves relative URLs"
    (is (= "https://example.com/other"
           (crawler/resolve-url "https://example.com/page" "/other")))))

(deftest test-crawler-creation
  (testing "make-crawler creates valid state"
    (let [c (crawler/make-crawler)]
      (is (set? (:visited c)))
      (is (empty? (:visited c)))
      (is (vector? (:frontier c)))
      (is (map? (:robots c)))
      (is (map? (:limiters c)))
      (is (map? (:config c)))
      (is (string? (get-in c [:config :user-agent])))))

  (testing "make-crawler accepts options"
    (let [c (crawler/make-crawler {:user-agent "TestBot/1.0" :default-delay-ms 1000})]
      (is (= "TestBot/1.0" (get-in c [:config :user-agent])))
      (is (= 1000 (get-in c [:config :default-delay-ms]))))))

(deftest test-frontier-management
  (testing "enqueue-urls! adds entries sorted by EFE score"
    (let [c (crawler/make-crawler)
          urls [{:url "https://a.com" :efe-score 0.5}
                {:url "https://b.com" :efe-score 0.1}
                {:url "https://c.com" :efe-score 0.9}]
          updated (crawler/enqueue-urls! c urls)]
      (is (= 3 (count (:frontier updated))))
      ;; Should be sorted by efe-score ascending
      (is (= 0.1 (:efe-score (first (:frontier updated)))))))

  (testing "enqueue-urls! filters visited URLs"
    (let [c (-> (crawler/make-crawler)
                (update :visited conj "https://a.com"))
          urls [{:url "https://a.com" :efe-score 0.1}
                {:url "https://b.com" :efe-score 0.2}]
          updated (crawler/enqueue-urls! c urls)]
      (is (= 1 (count (:frontier updated))))
      (is (= "https://b.com" (:url (first (:frontier updated)))))))

  (testing "next-url pops lowest EFE entry"
    (let [c (crawler/enqueue-urls! (crawler/make-crawler)
                                    [{:url "https://a.com" :efe-score 0.5}
                                     {:url "https://b.com" :efe-score 0.1}])
          [updated-c url] (crawler/next-url c)]
      (is (= "https://b.com" url))
      (is (= 1 (count (:frontier updated-c))))))

  (testing "next-url returns nil for empty frontier"
    (let [[_ url] (crawler/next-url (crawler/make-crawler))]
      (is (nil? url)))))

(deftest test-robots-txt-compliance
  (testing "allowed? returns true by default (no robots.txt)"
    (let [c (crawler/make-crawler)]
      (is (true? (crawler/allowed? c "https://example.com/page"))))))

(deftest test-frontier-size
  (let [c (crawler/enqueue-urls! (crawler/make-crawler)
                                  [{:url "https://a.com" :efe-score 0.1}
                                   {:url "https://b.com" :efe-score 0.2}])]
    (is (= 2 (crawler/frontier-size c)))
    (is (= 0 (crawler/visited-count c)))))
