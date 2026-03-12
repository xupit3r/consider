(ns consider.web.extractor
  "Content extraction from HTML using jsoup. Removes boilerplate,
   extracts clean text, links with context, and chunks text for LLM processing."
  (:require [clojure.string :as str])
  (:import [org.jsoup Jsoup]
           [org.jsoup.nodes Document Element TextNode]
           [org.jsoup.select Elements]))

;; --- Boilerplate Removal ---

(def ^:private boilerplate-selectors
  "CSS selectors for common boilerplate elements."
  ["nav" "header" "footer" "aside" ".sidebar" ".menu" ".navigation"
   ".cookie-banner" ".advertisement" ".ad" "#cookie-consent"
   "script" "style" "noscript" "iframe" ".social-share"
   ".comments" "#comments" ".related-posts"])

(defn- remove-boilerplate
  "Removes boilerplate elements from a parsed HTML document."
  [^Document doc]
  (doseq [selector boilerplate-selectors]
    (try
      (let [elements (.select doc selector)]
        (.remove elements))
      (catch Exception _)))
  doc)

;; --- Text Extraction ---

(defn- extract-main-text
  "Extracts main content text, preferring article/main elements."
  [^Document doc]
  (let [main-selectors ["article" "main" "[role=main]" ".content" ".post-content"
                        ".article-body" ".entry-content" "#content"]
        main-el (some (fn [sel]
                        (let [els (.select doc sel)]
                          (when-not (.isEmpty els)
                            (.first els))))
                      main-selectors)]
    (let [el (or main-el (.body doc))]
      (if el (or (.text el) "") ""))))

;; --- Link Extraction ---

(defn- extract-links
  "Extracts links with anchor text and surrounding context."
  [^Document doc base-url]
  (let [links (.select doc "a[href]")]
    (reduce (fn [acc ^Element link]
              (let [href (.absUrl link "href")
                    anchor (.text link)
                    ;; Get surrounding text context (parent paragraph or sentence)
                    parent (.parent link)
                    context (when parent
                              (let [parent-text (.text parent)]
                                (when (and (not (str/blank? parent-text))
                                           (<= (count parent-text) 500))
                                  parent-text)))]
                (if (and (not (str/blank? href))
                         (or (str/starts-with? href "http://")
                             (str/starts-with? href "https://"))
                         (not (str/ends-with? href ".jpg"))
                         (not (str/ends-with? href ".png"))
                         (not (str/ends-with? href ".gif"))
                         (not (str/ends-with? href ".pdf"))
                         (not (str/ends-with? href ".zip")))
                  (conj acc {:url href
                             :anchor-text (when-not (str/blank? anchor) anchor)
                             :context (or context "")})
                  acc)))
            []
            links)))

;; --- Text Chunking ---

(defn- chunk-text
  "Splits text into chunks suitable for LLM processing.
   Targets ~max-words per chunk."
  [text max-words]
  (if (or (nil? text) (str/blank? text))
    []
    (let [;; Split by lines first to avoid massive regex splits
          lines (str/split-lines text)
          ;; Remove blank lines
          lines (remove str/blank? lines)]
      (loop [remaining-lines lines
             current-chunk []
             current-words 0
             chunks []]
        (if (empty? remaining-lines)
          (if (empty? current-chunk)
            chunks
            (conj chunks (str/join "\n" current-chunk)))
          (let [line (first remaining-lines)
                line-words (count (str/split line #"\s+"))
                new-words (+ current-words line-words)]
            (if (and (> current-words 0) (> new-words max-words))
              ;; Start new chunk
              (recur (rest remaining-lines)
                     [line]
                     line-words
                     (conj chunks (str/join "\n" current-chunk)))
              ;; Add to current chunk
              (recur (rest remaining-lines)
                     (conj current-chunk line)
                     new-words
                     chunks))))))))

;; --- Main Extraction ---

(defn extract-content
  "Extracts structured content from an HTML string.
   Returns {:title :text :links [{:url :anchor-text :context}] :chunks [...]}"
  ([html url] (extract-content html url {}))
  ([html url opts]
   (if (str/blank? html)
     {:title nil :text "" :links [] :chunks []}
     (let [^Document doc (Jsoup/parse html url)
           _ (remove-boilerplate doc)
           text (extract-main-text doc)
           links (extract-links doc url)
           max-words (or (:max-chunk-words opts) 500)
           chunks (chunk-text text max-words)]
       {:title (when-not (str/blank? (.title doc)) (.title doc))
        :text text
        :links (vec (distinct links))
        :chunks chunks}))))

(defn extract-text-only
  "Lightweight extraction — just clean text, no link analysis."
  [html]
  (if (str/blank? html)
    ""
    (let [doc (Jsoup/parse html)
          _ (remove-boilerplate doc)]
      (extract-main-text doc))))

(defn extract-search-results
  "Extracts search result links from a search engine results page.
   Returns [{:url :title :snippet}]"
  [html search-engine url]
  (if (str/blank? html)
    []
    (let [^Document doc (Jsoup/parse html url)]
      (case search-engine
        :wikipedia
        (let [results (.select doc ".mw-search-result")]
          (mapv (fn [^Element el]
                  (let [link (.select el "a")
                        title (.text link)
                        href (.absUrl link "href")
                        snippet (.text (.select el ".mw-search-result-text"))]
                    {:url href :title title :snippet snippet}))
                results))

        :duckduckgo
        (let [results (.select doc ".result__a")]
          (mapv (fn [^Element el]
                  (let [title (.text el)
                        href (.absUrl el "href")
                        ;; Snippet is usually in the next element or nearby in DDGO HTML
                        snippet (try (.text (.nextElementSibling (.parent el)))
                                     (catch Exception _ ""))]
                    {:url href :title title :snippet snippet}))
                results))

        ;; Default: just extract all links
        (mapv (fn [l] {:url (:url l) :title (:anchor-text l) :snippet (:context l)})
              (extract-links doc url))))))
