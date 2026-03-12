# Spec 03: Crawler Hardening

## Priority: P1

## Current State

`src/consider/web/crawler.clj` works for basic cases. Tests pass. But it needs hardening for real-world use.

## Work Items

### 3a. Robots.txt caching expiry

Currently `ensure-robots` fetches robots.txt once per domain and caches forever. Add TTL-based expiry.

```clojure
;; In ensure-robots, check if cached robots are stale:
(defn- ensure-robots [crawler-state domain]
  (let [existing (get-in crawler-state [:robots domain])
        ttl-ms (* 24 60 60 1000)  ;; 24 hours
        stale? (and existing
                    (> (- (System/currentTimeMillis) (:fetched-at existing)) ttl-ms))]
    (if (and existing (not stale?))
      crawler-state
      (let [robots (fetch-robots-txt domain (:config crawler-state))]
        (assoc-in crawler-state [:robots domain] robots)))))
```

### 3b. URL normalization edge cases

`normalize-url` doesn't handle:
- Case normalization of scheme/host (`HTTP://Example.COM` -> `https://example.com`)
- Default port removal (`:80` for http, `:443` for https)
- Percent-encoding normalization

Add these and add tests.

### 3c. Content-type filtering in `fetch-page`

Currently `fetch-page` accepts any response. It should:
- Check `Content-Type` header — only process `text/html` and `application/xhtml+xml`
- Return nil for binary content (PDFs, images, etc.)
- Respect `Content-Length` limits (reject responses > 10MB)

### 3d. Redirect tracking

`fetch-page` uses `:redirect-strategy :lax` but doesn't record the final URL after redirects. The visited set should include the final URL, not just the requested one. Use the response's `:uri` or final URL.

## Files

- `src/consider/web/crawler.clj`
- `test/consider/web/crawler_test.clj`

## Acceptance Criteria

- All existing crawler tests still pass
- New tests for each hardening item
- No network calls in unit tests (mock HTTP where needed, or test only the pure functions)
