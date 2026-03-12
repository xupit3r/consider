# Neanderthal Performance Patterns

To achieve near-native performance in Clojure using Neanderthal, follow these established patterns.

## 1. In-Place Operations (The '!' Rule)

Allocation is the primary enemy of performance in high-speed math. Always prefer destructive functions that mutate an existing matrix.

```clojure
(require '[uncomplicate.neanderthal.core :refer [mm! scal! axpy!]])

;; BAD: Allocates a new matrix for the result
(let [c (mm a b)] ...)

;; GOOD: Reuses matrix 'c'
(mm! a b c)
```

## 2. Choosing the Right Backend

- **Native (`uncomplicate.neanderthal.native`)**: Best for CPU-bound tasks. Uses MKL or OpenBLAS.
- **CUDA (`uncomplicate.neanderthal.cuda`)**: Best for massively parallel tasks (large matrices).
- **OpenCL (`uncomplicate.neanderthal.opencl`)**: Cross-platform GPU support.

## 3. Element-wise Operations

Avoid standard Clojure `map` on matrices; it boxes primitives and destroys performance.

```clojure
;; Fast element-wise function mapping
(fmap! (fn [^double x] (Math/exp x)) my-matrix)

;; Scalar addition (y = ax + y)
(axpy! 1.0 ones-vector my-vector)
```

## 4. Matrix Decompositions

Neanderthal provides high-level wrappers for LAPACK routines.

- **`svd!`**: Singular Value Decomposition.
- **`evd!`**: Eigenvalue Decomposition.
- **`lu!` / `trsv!`**: Solving linear systems.

## 5. Layout and Transposition

BLAS/LAPACK is **Column-Major**. If your data source is row-major (like NumPy or some C libs), specify it in the constructor:

```clojure
(dge 10 10 {:layout :row})
```
Avoid calling `trans` frequently; most functions like `mm!` have flags to handle transposition on the fly:
`(mm! 1.0 (trans a) b 0.0 c)`
