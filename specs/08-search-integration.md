# Spec 08: Search Integration & Goal-Directed Mode

## Priority: P2

## Current State

The forager can crawl from seed URLs and follow links. But it cannot:
- Issue web searches to discover new seed URLs
- Bias crawling toward specific knowledge goals

## Work Items

### 8a. Search URL construction

When the MCTS selects a `SEARCH:query` action, the forager needs to turn it into fetchable URLs. Options:

1. **Wikipedia search**: `https://en.wikipedia.org/w/index.php?search=QUERY&title=Special:Search`
2. **Google Scholar** (via MCP): use the `google-scholar` MCP server already configured
3. **DuckDuckGo HTML**: `https://html.duckduckgo.com/html/?q=QUERY` (no JS needed, parseable with jsoup)

Implement at least option 1 (Wikipedia search) and option 3 (DuckDuckGo):

```clojure
(defn search-wikipedia [query]
  "Returns a URL for Wikipedia search results page."
  (str "https://en.wikipedia.org/w/index.php?search="
       (java.net.URLEncoder/encode query "UTF-8")
       "&title=Special:Search"))

(defn search-duckduckgo [query]
  "Returns a URL for DuckDuckGo HTML search results."
  (str "https://html.duckduckgo.com/html/?q="
       (java.net.URLEncoder/encode query "UTF-8")))
```

### 8b. Search result parsing

Add to `extractor.clj`:

```clojure
(defn extract-search-results
  "Extracts search result links from a search engine results page.
   Returns [{:url :title :snippet}]"
  [html search-engine]
  ...)
```

Handle at least:
- Wikipedia search results (parse `.mw-search-result` elements)
- DuckDuckGo HTML results (parse `.result__a` elements)

### 8c. Goal-directed EFE bias

Currently `score-url` gives a goal-relevance bonus based on keyword matching. Enhance it:

1. **Term frequency**: Weight matches by how rare the goal term is (TF-IDF-like)
2. **Graph distance**: URLs that mention entities close to goal entities in the graph get lower EFE
3. **Recency**: Recently discovered URLs about the goal topic get priority

### 8d. Forager mode switching

Add explicit mode to forager state:

```clojure
{:mode :curiosity}  ;; default — EFE ambiguity dominates
{:mode :goal-directed :goal "causes of World War 1"}  ;; EFE risk dominates
```

In `score-url`, adjust the risk/ambiguity weighting:
- `:curiosity` mode: `(+ (* 0.3 risk) (* 0.7 ambiguity))`
- `:goal-directed` mode: `(+ (* 0.8 risk) (* 0.2 ambiguity))`

### 8e. Action dispatch in forager

Update `forage-step` (or add a wrapper) to handle the parsed action from `core/step`:

```clojure
(defn execute-action
  "Dispatches a parsed action. Returns [updated-forager-state observation-vector]."
  [forager-state action]
  (case (:type action)
    :visit  (forage-step forager-state :url (:url action))
    :search (let [search-url (search-wikipedia (:query action))]
              (forage-step forager-state :url search-url))
    ;; default: pick from frontier
    (forage-step forager-state)))
```

## Files

- `src/consider/web/crawler.clj` (search URL construction)
- `src/consider/web/extractor.clj` (search result parsing)
- `src/consider/web/forager.clj` (mode switching, action dispatch)
- New tests for each

## Acceptance Criteria

- `SEARCH:query` actions produce fetchable URLs
- Wikipedia and DuckDuckGo search results are parsed into links
- Goal-directed mode demonstrably biases crawling toward the goal topic
- Test: set goal "causes of World War 1", run 5 steps from Wikipedia seed, verify frontier URLs are relevant
