(ns consider.models
  "Neural model definitions using Neanderthal (uncomplicate.neanderthal)."
  (:require [uncomplicate.neanderthal.core :as n]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.random :as rng]
            [uncomplicate.neanderthal.vect-math :as vm]
            [uncomplicate.neanderthal.real :as r]))

(defrecord NeanderthalMLP [w1 b1 w2 b2 state-dim obs-dim hidden-dim])

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
    (->NeanderthalMLP w1 b1 w2 b2 state-dim observation-dim hidden-dim)))

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
  (let [{:keys [w1 b1 w2 b2 state-dim obs-dim]} net
        actual-obs-dim (n/dim observation)
        actual-state-dim (n/dim x)
        ;; We use the dimensions the network was TRAINED for to construct the input
        input-dim (+ state-dim 1 obs-dim)
        input (native/dv input-dim)]
    ;; Construct input: [x, t, observation] 
    ;; Use min to handle potential mismatches during growth phases safely
    (dotimes [i (min state-dim actual-state-dim)]
      (n/entry! input i (n/entry x i)))

    (let [t-idx state-dim]
      (n/entry! input t-idx (double t)))

    (dotimes [i (min obs-dim actual-obs-dim)]
      (n/entry! input (+ state-dim 1 i) (n/entry observation i)))

    ;; Layer 1
    (let [h1 (n/mv w1 input)]
      (n/axpy! 1.0 b1 h1)
      (relu! h1)
      ;; Layer 2
      (let [out-v (n/mv w2 h1)]
        (n/axpy! 1.0 b2 out-v)

        ;; Return a vector matching the ACTUAL state dim of the particles
        (let [final-out (native/dv actual-state-dim)]
          (n/scal! 0.0 final-out)
          (dotimes [i (min actual-state-dim (n/dim out-v))]
            (let [v (n/entry out-v i)]
              (n/entry! final-out i (max -10.0 (min 10.0 v)))))
          final-out)))))

(defn train-batch!
  "Trains the Neanderthal MLP using Mini-Batch SGD.
   input-mat: [BatchSize x InputDim] matrix.
   target-mat: [BatchSize x StateDim] matrix.
   Uses Matrix-Matrix operations for high performance.
   Includes gradient clipping for stability."
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
            _ (dotimes [i batch-size] (n/axpy! 1.0 (n/row dh1-pre i) db1))

            ;; Gradient Clipping
            clip-val 5.0
            clip-fn (fn [m]
                      (let [norm (r/nrm2 m)]
                        (when (or (Double/isNaN norm) (> norm clip-val))
                          (n/scal! (/ clip-val (or (and (not (zero? norm)) norm) 1.0)) m))))]

        (clip-fn dw1)
        (clip-fn db1)
        (clip-fn dw2)
        (clip-fn db2)

        ;; Update weights
        (n/axpy! (- learning-rate) dw1 w1)
        (n/axpy! (- learning-rate) db1 b1)
        (n/axpy! (- learning-rate) dw2 w2)
        (n/axpy! (- learning-rate) db2 b2)))))

(defn grow-network
  "Expands the input and output layers of the MLP to handle new slots.
   Preserves existing weights."
  [net new-state-dim new-obs-dim]
  (let [{:keys [w1 b1 w2 b2 state-dim obs-dim hidden-dim]} net
        new-input-dim (+ new-state-dim 1 new-obs-dim)

        new-w1 (native/dge hidden-dim new-input-dim)
        new-w2 (native/dge new-state-dim hidden-dim)
        new-b2 (native/dv new-state-dim)]

    ;; Initialize new weights with small noise
    (rng/rand-normal! 0.0 0.01 new-w1)
    (rng/rand-normal! 0.0 0.01 new-w2)
    (n/scal! 0.0 new-b2)

    ;; 1. Copy State weights [0...state-dim-1]
    (dotimes [i hidden-dim]
      (dotimes [j (min state-dim new-state-dim)]
        (n/entry! new-w1 i j (n/entry w1 i j))))

    ;; 2. Copy Time weight [state-dim] -> [new-state-dim]
    (dotimes [i hidden-dim]
      (n/entry! new-w1 i new-state-dim (n/entry w1 i state-dim)))

    ;; 3. Copy Observation weights [state-dim+1...end] -> [new-state-dim+1...end]
    (dotimes [i hidden-dim]
      (dotimes [j (min obs-dim new-obs-dim)]
        (n/entry! new-w1 i (+ new-state-dim 1 j) (n/entry w1 i (+ state-dim 1 j)))))

    ;; 4. Copy Output weights
    (dotimes [i (min state-dim new-state-dim)]
      (dotimes [j hidden-dim]
        (n/entry! new-w2 i j (n/entry w2 i j))))

    (dotimes [i (min state-dim new-state-dim)]
      (n/entry! new-b2 i (n/entry b2 i)))

    (assoc net
           :w1 new-w1 :w2 new-w2 :b2 new-b2
           :state-dim new-state-dim :obs-dim new-obs-dim)))
