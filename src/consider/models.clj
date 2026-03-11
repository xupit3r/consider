(ns consider.models
  "Neural model definitions using Neanderthal (uncomplicate.neanderthal)."
  (:require [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.random :as rng]
            [uncomplicate.neanderthal.vect-math :as vm]))

(defrecord NeanderthalMLP [w1 b1 w2 b2])

(defn make-mlp-vector-field
  "Creates a simple 2-layer MLP for flow matching using Neanderthal.
   Input: [state-dim + 1 (time) + observation-dim]
   Output: [state-dim]"
  [state-dim observation-dim hidden-dim]
  (let [input-dim (+ state-dim 1 observation-dim)
        w1 (native/dge hidden-dim input-dim)
        b1 (native/dv hidden-dim)
        w2 (native/dge state-dim hidden-dim)
        b2 (native/dv state-dim)]
    ;; Initialize weights (Xavier-like)
    (rng/rand-normal! 0.0 (/ 1.0 (Math/sqrt input-dim)) w1)
    (rng/rand-normal! 0.0 (/ 1.0 (Math/sqrt hidden-dim)) w2)
    (n/scal! 0.0 b1)
    (n/scal! 0.0 b2)
    (->NeanderthalMLP w1 b1 w2 b2)))

(defn relu! [v]
  (if (n/vctr? v)
    (dotimes [i (n/dim v)]
      (n/entry! v i (max 0.0 (n/entry v i))))
    (dotimes [i (n/mrows v)]
      (dotimes [j (n/ncols v)]
        (n/entry! v i j (max 0.0 (n/entry v i j))))))
  v)

(defn relu-grad [v]
  (let [res (n/copy v)]
    (if (n/vctr? v)
      (dotimes [i (n/dim res)]
        (n/entry! res i (if (> (n/entry v i) 0.0) 1.0 0.0)))
      (dotimes [i (n/mrows res)]
        (dotimes [j (n/ncols res)]
          (n/entry! res i j (if (> (n/entry v i j) 0.0) 1.0 0.0)))))
    res))

(defn predict-velocity
  "Uses the Neanderthal MLP to predict velocity.
   Includes output clipping for numerical stability in continuous integration."
  [net x t observation]
  (let [{:keys [w1 b1 w2 b2]} net
        obs-dim (n/dim observation)
        state-dim (n/dim x)
        input-dim (+ state-dim 1 obs-dim)
        input (native/dv input-dim)]
    ;; Construct input: [x, t, observation]
    (dotimes [i state-dim] (n/entry! input i (n/entry x i)))
    (n/entry! input state-dim (double t))
    (dotimes [i obs-dim] (n/entry! input (+ state-dim 1 i) (n/entry observation i)))

    ;; Layer 1
    (let [h1 (n/mv w1 input)]
      (n/axpy! 1.0 b1 h1)
      (relu! h1)
      ;; Layer 2
      (let [out (n/mv w2 h1)]
        (n/axpy! 1.0 b2 out)
        ;; Clip output to prevent exploding gradients/steps
        (dotimes [i (n/dim out)]
          (let [v (n/entry out i)]
            (n/entry! out i (max -10.0 (min 10.0 v)))))
        out))))

(defn train-batch!
  "Trains the Neanderthal MLP using Mini-Batch SGD.
   input-mat: [BatchSize x InputDim] matrix.
   target-mat: [BatchSize x StateDim] matrix.
   Uses Matrix-Matrix operations for high performance."
  [net input-mat target-mat learning-rate iterations]
  (let [{:keys [w1 b1 w2 b2]} net
        batch-size (n/mrows input-mat)]
    (dotimes [_ iterations]
      (let [;; Forward pass
            h1-pre (n/mm input-mat (n/trans w1))
            _ (dotimes [i batch-size]
                (n/axpy! 1.0 b1 (n/row h1-pre i)))
            h1 (relu! (n/copy h1-pre))

            out (n/mm h1 (n/trans w2))
            _ (dotimes [i batch-size]
                (n/axpy! 1.0 b2 (n/row out i)))

            ;; Loss gradient
            dout (n/copy out)
            _ (n/axpy! -1.0 target-mat dout)
            _ (n/scal! (/ 2.0 batch-size) dout)

            ;; Backprop
            dw2 (n/mm (n/trans dout) h1)
            db2 (native/dv (n/ncols dout))
            _ (n/scal! 0.0 db2)
            _ (dotimes [i batch-size] (n/axpy! 1.0 (n/row dout i) db2))

            dh1 (n/mm dout w2)
            dh1-pre (vm/mul dh1 (relu-grad h1-pre))

            dw1 (n/mm (n/trans dh1-pre) input-mat)
            db1 (native/dv (n/mrows w1))
            _ (n/scal! 0.0 db1)
            _ (dotimes [i batch-size] (n/axpy! 1.0 (n/row dh1-pre i) db1))]

        ;; Update weights
        (n/axpy! (- learning-rate) dw1 w1)
        (n/axpy! (- learning-rate) db1 b1)
        (n/axpy! (- learning-rate) dw2 w2)
        (n/axpy! (- learning-rate) db2 b2)))))

(defn- copy-into!
  "Helper to copy data from a smaller structure into a larger one at specified offsets."
  [src dest row-offset col-offset]
  (if (n/vctr? src)
    (dotimes [i (n/dim src)]
      (n/entry! dest (+ row-offset i) (n/entry src i)))
    (dotimes [i (n/mrows src)]
      (dotimes [j (n/ncols src)]
        (n/entry! dest (+ row-offset i) (+ col-offset j) (n/entry src i j)))))
  dest)

(defn grow-network
  "Expands the input and output layers of the MLP to handle new slots.
   Preserves existing weights."
  [net new-state-dim new-obs-dim]
  (let [{:keys [w1 b1 w2 b2]} net
        hidden-dim (n/mrows w1)
        old-input-dim (n/ncols w1)
        old-state-dim (n/mrows w2)
        old-obs-dim (- old-input-dim old-state-dim 1)

        new-input-dim (+ new-state-dim 1 new-obs-dim)

        new-w1 (native/dge hidden-dim new-input-dim)
        new-w2 (native/dge new-state-dim hidden-dim)
        new-b2 (native/dv new-state-dim)]

    (rng/rand-normal! 0.0 0.01 new-w1)
    (rng/rand-normal! 0.0 0.01 new-w2)
    (n/scal! 0.0 new-b2)

    (dotimes [i hidden-dim]
      (dotimes [j old-state-dim]
        (n/entry! new-w1 i j (n/entry w1 i j))))
    (dotimes [i hidden-dim]
      (n/entry! new-w1 i new-state-dim (n/entry w1 i old-state-dim)))
    (dotimes [i hidden-dim]
      (dotimes [j old-obs-dim]
        (n/entry! new-w1 i (+ new-state-dim 1 j) (n/entry w1 i (+ old-state-dim 1 j)))))

    (copy-into! w2 new-w2 0 0)
    (copy-into! b2 new-b2 0 0)

    (assoc net :w1 new-w1 :w2 new-w2 :b2 new-b2)))
