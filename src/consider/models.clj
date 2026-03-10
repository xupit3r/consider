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
  (dotimes [i (n/dim v)]
    (n/entry! v i (max 0.0 (n/entry v i))))
  v)

(defn relu-grad [v]
  (let [res (n/copy v)]
    (dotimes [i (n/dim res)]
      (n/entry! res i (if (> (n/entry v i) 0.0) 1.0 0.0)))
    res))

(defn predict-velocity
  "Uses the Neanderthal MLP to predict velocity."
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
        out))))

(defn train-on-samples!
  "Trains the Neanderthal MLP using SGD.
   Simplified: Single sample training."
  [net input-v target-v learning-rate iterations]
  (let [{:keys [w1 b1 w2 b2]} net]
    (dotimes [_ iterations]
      (let [state-dim (n/mrows w2)
            hidden-dim (n/mrows w1)
            input-dim (n/ncols w1)
            
            ;; Forward pass
            h1-pre (n/mv w1 input-v)
            _ (n/axpy! 1.0 b1 h1-pre)
            h1 (relu! (n/copy h1-pre))
            
            out (n/mv w2 h1)
            _ (n/axpy! 1.0 b2 out)
            
            ;; Loss gradient (quadratic loss: (out - target)^2)
            ;; dLoss/dout = 2 * (out - target)
            dout (n/copy out)
            _ (n/axpy! -1.0 target-v dout)
            _ (n/scal! 2.0 dout)
            
            ;; Backprop to w2, b2
            ;; dw2 = dout * h1^T
            dw2 (n/rk dout h1)
            db2 (n/copy dout)
            
            ;; Backprop to h1
            ;; dh1 = w2^T * dout
            dh1 (n/mv (n/trans w2) dout)
            ;; dh1_pre = dh1 * relu'(h1_pre)
            dh1-pre (vm/mul dh1 (relu-grad h1-pre))
            
            ;; Backprop to w1, b1
            dw1 (n/rk dh1-pre input-v)
            db1 (n/copy dh1-pre)]
        
        ;; Update weights
        (n/axpy! (- learning-rate) dw1 w1)
        (n/axpy! (- learning-rate) db1 b1)
        (n/axpy! (- learning-rate) dw2 w2)
        (n/axpy! (- learning-rate) db2 b2)))))
