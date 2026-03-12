# Spec 01: Fix extractor.clj NPE

## Priority: P0 (blocks 3 tests)

## Problem

`consider.web.extractor/extract-content` throws a NullPointerException in `chunk-text` when processing HTML with whitespace-heavy formatting. The error is:

```
java.lang.NullPointerException: Cannot invoke "java.lang.CharSequence.length()" because "this.text" is null
  at java.util.regex.Pattern.split
  at clojure.string/split
  at consider.web.extractor/chunk-text (extractor.clj:93-95)
```

## Root Cause

The `chunk-text` function receives a non-nil, non-blank `text` string (confirmed via manual debugging), yet `(first paras)` returns `nil` during iteration. The defensive `(or para "")` guard was added but the bug persists — the exact mechanism needs investigation.

**Key clue**: The bug reproduces with whitespace-heavy HTML (indented body tags) but NOT with compact single-line HTML. The text returned by `extract-main-text` is fine in both cases — the problem is somewhere in `chunk-text`'s lazy-seq iteration.

## Reproducer

```clojure
(require '[consider.web.extractor :as e])

;; This WORKS:
(e/extract-content "<html><body><main><p>Hello world.</p></main></body></html>" "https://example.com")

;; This FAILS with NPE:
(e/extract-content "<html><body>
  <nav>Navigation</nav>
  <main><p>Main content here.</p></main>
  <footer>Footer stuff</footer>
  <script>alert('x')</script>
</body></html>" "https://example.com")
```

## File

`src/consider/web/extractor.clj` — the `chunk-text` function (line 76-106)

## Approach

1. Reproduce the bug in a REPL
2. Add `(println)` or step through `chunk-text` to find exactly which call to `str/split` gets a nil argument
3. The likely fix is one of:
   - The `text` variable is actually nil/null (Java null, not Clojure nil) in some edge case from jsoup's `.text` method
   - The `str/split #"\n\n+"` is producing nil entries in the resulting vector
   - The lazy seq from `remove` is misbehaving (try `(vec (remove ...))` to force eagerness)
4. Fix and verify all 3 failing tests pass:
   - `test-extract-content`
   - `test-boilerplate-removal`
   - `test-link-filtering`

## Acceptance Criteria

- All tests in `consider.web.extractor-test` pass
- `extract-content` handles arbitrary HTML without NPE
- No changes to the public API
