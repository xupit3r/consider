# Neanderthal & dtype-next Memory Management

Native-speed computing requires explicit memory management to avoid GC pressure and out-of-memory errors on off-heap data.

## 1. Zero-Copy Interoperability with `dtype-next`

`dtype-next` allows you to wrap off-heap data from other sources (databases, files) as Neanderthal matrices without copying.

```clojure
(require '[tech.v3.datatype :as dtype]
         '[uncomplicate.neanderthal.native :refer [dge]])

;; Load or create off-heap native memory
(let [native-buf (dtype/->native-heap [1.0 2.0 3.0])
      ;; Create a view as a Neanderthal vector
      vec (dv native-buf)]
  ...)
```

## 2. Manual Release (The 'with-release' Pattern)

Because Neanderthal uses off-heap `DirectByteBuffers`, the JVM's Garbage Collector cannot immediately free the memory. Use `uncomplicate.commons.core/with-release` for deterministic cleanup.

```clojure
(require '[uncomplicate.commons.core :refer [with-release]])

(with-release [a (dge 1000 1000)]
  ;; Perform heavy math
  (let [res (do-something a)]
    ;; 'a' will be released when this block finishes
    res))
```

## 3. Buffer Slicing and Views

Avoid copying when you only need a subset of a matrix or vector.

- **`submatrix`**: Returns a view of a rectangular subset.
- **`subvector`**: Returns a view of a contiguous segment.
- **`view-ge` / `view-tr`**: Change the interpretation of the underlying buffer (e.g., viewing a flat buffer as a matrix).

## 4. Avoiding Boxing in Implementation

- Always type-hint functions that handle Neanderthal types.
- Avoid passing Neanderthal objects into generic Clojure collections (sets, maps) unless necessary.
- Use `uncomplicate.neanderthal.core/fold` for primitive-speed reductions.
