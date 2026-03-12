(ns consider.web.crawler
  "Web fetching with robots.txt compliance, rate limiting, and crawl frontier management."
  (:require [hato.client :as http]
            [clojure.string :as str])
  (:import [java.net URL URI]
           [java.util.concurrent ConcurrentHashMap]))

;; --- URL Utilities ---

(defn normalize-url
  "Normalizes a URL by removing fragments and trailing slashes."
  [url-str]
  (try
    (let [uri (URI. url-str)
          normalized (URI. (.getScheme uri)
                           (.getAuthority uri)
                           (.getPath uri)
                           (.getQuery uri)
                           nil)]
      (-> (.toString normalized)
          (str/replace #"/+$" "")
          (str/replace #"^(https?://[^/]+)$" "$1")))
    (catch Exception _ url-str)))

(defn extract-domain
  "Extracts the domain (host) from a URL string."
  [url-str]
  (try
    (.getHost (URL. url-str))
    (catch Exception _ nil)))

(defn resolve-url
  "Resolves a potentially relative URL against a base URL."
  [base-url relative-url]
  (try
    (let [base (URI. base-url)
          resolved (.resolve base relative-url)]
      (.toString resolved))
    (catch Exception _ nil)))

(defn search-wikipedia
  "Returns a URL for Wikipedia search results page."
  [query]
  (str "https://en.wikipedia.org/w/index.php?search="
       (java.net.URLEncoder/encode query "UTF-8")
       "&title=Special:Search"))

(defn search-duckduckgo
  "Returns a URL for DuckDuckGo HTML search results."
  [query]
  (str "https://html.duckduckgo.com/html/?q="
       (java.net.URLEncoder/encode query "UTF-8")))

;; --- Robots.txt ---

(defn- parse-robots-txt
  "Parses robots.txt content into a map of user-agent -> rules."
  [content]
  (let [lines (str/split-lines (or content ""))
        ;; Simple parser: collect Disallow/Allow rules per user-agent
        result (atom {:default {:disallow [] :allow [] :crawl-delay nil}})]
    (loop [lines lines
           current-agent :default]
      (if (empty? lines)
        @result
        (let [line (str/trim (first lines))
              line (str/replace line #"#.*$" "")]
          (cond
            (str/blank? line)
            (recur (rest lines) current-agent)

            (str/starts-with? (str/lower-case line) "user-agent:")
            (let [agent (str/trim (subs line (count "user-agent:")))]
              (swap! result assoc-in [(keyword agent) :disallow] [])
              (swap! result assoc-in [(keyword agent) :allow] [])
              (recur (rest lines) (keyword agent)))

            (str/starts-with? (str/lower-case line) "disallow:")
            (let [path (str/trim (subs line (count "disallow:")))]
              (when-not (str/blank? path)
                (swap! result update-in [current-agent :disallow] conj path))
              (recur (rest lines) current-agent))

            (str/starts-with? (str/lower-case line) "allow:")
            (let [path (str/trim (subs line (count "allow:")))]
              (when-not (str/blank? path)
                (swap! result update-in [current-agent :allow] conj path))
              (recur (rest lines) current-agent))

            (str/starts-with? (str/lower-case line) "crawl-delay:")
            (let [delay-str (str/trim (subs line (count "crawl-delay:")))]
              (try
                (swap! result assoc-in [current-agent :crawl-delay]
                       (long (* 1000 (Double/parseDouble delay-str))))
                (catch Exception _))
              (recur (rest lines) current-agent))

            :else
            (recur (rest lines) current-agent)))))))

(defn fetch-robots-txt
  "Fetches and parses robots.txt for a domain."
  [domain config]
  (try
    (let [url (str "https://" domain "/robots.txt")
          resp (http/get url {:headers {"User-Agent" (:user-agent config)}
                              :timeout 5000
                              :as :string})]
      (if (= 200 (:status resp))
        {:rules (parse-robots-txt (:body resp))
         :fetched-at (System/currentTimeMillis)}
        {:rules {:default {:disallow [] :allow [] :crawl-delay nil}}
         :fetched-at (System/currentTimeMillis)}))
    (catch Exception _
      {:rules {:default {:disallow [] :allow [] :crawl-delay nil}}
       :fetched-at (System/currentTimeMillis)})))

(defn- path-matches-pattern?
  "Checks if a URL path matches a robots.txt pattern."
  [path pattern]
  (cond
    (str/blank? pattern) false
    (= pattern "/") true
    (str/ends-with? pattern "*")
    (str/starts-with? path (subs pattern 0 (dec (count pattern))))
    :else
    (str/starts-with? path pattern)))

(defn allowed?
  "Checks if a URL is allowed by robots.txt rules."
  [crawler-state url]
  (let [domain (extract-domain url)
        robots (get-in crawler-state [:robots domain])
        rules (or (get-in robots [:rules :default])
                  {:disallow [] :allow []})
        path (try (.getPath (URL. url)) (catch Exception _ "/"))]
    ;; Allow rules take precedence over disallow
    (if (some #(path-matches-pattern? path %) (:allow rules))
      true
      (not (some #(path-matches-pattern? path %) (:disallow rules))))))

;; --- Rate Limiting ---

(defn- should-wait?
  "Returns milliseconds to wait before fetching from domain, or 0 if ready."
  [crawler-state domain]
  (let [limiter (get-in crawler-state [:limiters domain])
        delay-ms (or (:delay-ms limiter)
                     (get-in crawler-state [:robots domain :rules :default :crawl-delay])
                     (get-in crawler-state [:config :default-delay-ms]))
        last-ms (or (:last-ms limiter) 0)
        elapsed (- (System/currentTimeMillis) last-ms)]
    (if (>= elapsed delay-ms)
      0
      (- delay-ms elapsed))))

;; --- Crawler State ---

(defn make-crawler
  "Creates a new crawler state."
  ([] (make-crawler {}))
  ([opts]
   {:visited #{}
    :frontier []
    :robots {}
    :limiters {}
    :config {:user-agent (or (:user-agent opts) "Consider/0.1 (Active Inference Agent)")
             :default-delay-ms (or (:default-delay-ms opts) 2000)
             :max-per-domain (or (:max-per-domain opts) 100)}}))

(defn- ensure-robots
  "Ensures robots.txt is fetched for the domain."
  [crawler-state domain]
  (if (get-in crawler-state [:robots domain])
    crawler-state
    (let [robots (fetch-robots-txt domain (:config crawler-state))]
      (assoc-in crawler-state [:robots domain] robots))))

(defn fetch-page
  "Fetches a URL, respecting robots.txt and rate limits.
   Returns [updated-crawler-state {:status :body :url :content-type}] or [state nil] on failure."
  [crawler-state url]
  (let [domain (extract-domain url)
        state-with-robots (ensure-robots crawler-state domain)]
    (if-not (allowed? state-with-robots url)
      [state-with-robots nil]
      (let [wait-ms (should-wait? state-with-robots domain)]
        (when (pos? wait-ms)
          (Thread/sleep wait-ms))
        (try
          (let [resp (http/get url {:headers {"User-Agent" (get-in state-with-robots [:config :user-agent])}
                                    :timeout 10000
                                    :as :string
                                    :redirect-strategy :lax})
                updated-state (-> state-with-robots
                                  (update :visited conj (normalize-url url))
                                  (assoc-in [:limiters domain]
                                            {:last-ms (System/currentTimeMillis)
                                             :delay-ms (or (get-in state-with-robots [:config :default-delay-ms]) 2000)}))]
            [updated-state {:status (:status resp)
                            :body (:body resp)
                            :url url
                            :content-type (get-in resp [:headers "content-type"])}])
          (catch Exception e
            [(update state-with-robots :visited conj (normalize-url url))
             {:status 0 :body nil :url url :error (.getMessage e)}]))))))

(defn enqueue-urls!
  "Adds URLs to the frontier with EFE scores, filtering already-visited and disallowed URLs."
  [crawler-state urls-with-scores]
  (let [visited (:visited crawler-state)
        max-per-domain (get-in crawler-state [:config :max-per-domain])
        ;; Count existing per-domain
        domain-counts (frequencies (map #(extract-domain (:url %)) (:frontier crawler-state)))
        new-entries (remove
                     (fn [{:keys [url]}]
                       (let [norm (normalize-url url)
                             domain (extract-domain url)]
                         (or (contains? visited norm)
                             (>= (get domain-counts domain 0) max-per-domain)
                             (not (allowed? crawler-state url)))))
                     urls-with-scores)]
    (update crawler-state :frontier
            (fn [frontier]
              (sort-by :efe-score (into frontier new-entries))))))

(defn next-url
  "Returns [updated-crawler-state next-url] — pops the lowest-EFE URL from the frontier."
  [crawler-state]
  (if (empty? (:frontier crawler-state))
    [crawler-state nil]
    (let [best (first (:frontier crawler-state))
          updated (update crawler-state :frontier #(vec (rest %)))]
      [updated (:url best)])))

(defn frontier-size
  "Returns the number of URLs in the frontier."
  [crawler-state]
  (count (:frontier crawler-state)))

(defn visited-count
  "Returns the number of visited URLs."
  [crawler-state]
  (count (:visited crawler-state)))
