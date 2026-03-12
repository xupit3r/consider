# Spec 02: Fix forager.clj Return Value Bug

## Priority: P0 (incorrect return)

## Problem

In `consider.web.forager/forage-step`, there's a stale expression before the actual return:

```clojure
;; line 181-182 in forager.clj
obs-vector          ;; <-- bare expression, evaluated but discarded
[updated-forager obs-vector]))))))))
```

The bare `obs-vector` on line 181 does nothing (Clojure evaluates it and throws away the result). The function still returns `[updated-forager obs-vector]` from line 182. This isn't a correctness bug per se, but it's confusing dead code that suggests the author was undecided about the return shape.

## File

`src/consider/web/forager.clj` — `forage-step` function, around line 180-182

## Fix

Remove the bare `obs-vector` expression on line 181. The function should only return `[updated-forager obs-vector]`.

## Acceptance Criteria

- Dead expression removed
- `forage-step` returns `[updated-forager-state observation-vector]` as documented
- All tests in `consider.web.forager-test` still pass
