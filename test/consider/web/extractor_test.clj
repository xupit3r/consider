(ns consider.web.extractor-test
  (:require [clojure.test :refer :all]
            [consider.web.extractor :as extractor]))

(def sample-html
  "<html>
   <head><title>Active Inference - Wikipedia</title></head>
   <body>
     <nav><a href='/main'>Home</a></nav>
     <article>
       <h1>Active Inference</h1>
       <p>Active inference is a theory in neuroscience that describes how the brain
          works by minimizing free energy. It was developed by Karl Friston at UCL.</p>
       <p>The theory builds on the free energy principle and variational inference.
          It has applications in robotics, AI, and cognitive science.</p>
       <p>Key concepts include:</p>
       <ul>
         <li><a href='https://en.wikipedia.org/wiki/Free_energy_principle'>Free Energy Principle</a></li>
         <li><a href='https://en.wikipedia.org/wiki/Variational_inference'>Variational Inference</a></li>
         <li><a href='https://en.wikipedia.org/wiki/Karl_Friston'>Karl Friston</a></li>
       </ul>
       <p>Active inference extends predictive coding by adding action selection
          through expected free energy minimization.</p>
     </article>
     <footer>Copyright 2024</footer>
     <script>console.log('tracking');</script>
   </body>
   </html>")

(deftest test-extract-content
  (testing "extracts title"
    (let [result (extractor/extract-content sample-html "https://en.wikipedia.org/wiki/Active_Inference")]
      (is (= "Active Inference - Wikipedia" (:title result)))))

  (testing "extracts clean text (no nav/footer/script)"
    (let [result (extractor/extract-content sample-html "https://en.wikipedia.org/wiki/Active_Inference")
          text (:text result)]
      (is (clojure.string/includes? text "Active inference is a theory"))
      (is (clojure.string/includes? text "Karl Friston"))
      ;; Should not contain nav or footer content
      (is (not (clojure.string/includes? text "Copyright 2024")))
      (is (not (clojure.string/includes? text "tracking")))))

  (testing "extracts links with anchor text"
    (let [result (extractor/extract-content sample-html "https://en.wikipedia.org/wiki/Active_Inference")
          links (:links result)]
      (is (>= (count links) 3))
      (is (some #(= "https://en.wikipedia.org/wiki/Free_energy_principle" (:url %)) links))
      (is (some #(= "Karl Friston" (:anchor-text %)) links))))

  (testing "chunks text"
    (let [result (extractor/extract-content sample-html "https://en.wikipedia.org/wiki/Active_Inference")]
      (is (vector? (:chunks result)))
      (is (pos? (count (:chunks result)))))))

(deftest test-extract-empty
  (testing "handles empty/nil HTML"
    (let [result (extractor/extract-content "" "https://example.com")]
      (is (nil? (:title result)))
      (is (= "" (:text result)))
      (is (empty? (:links result)))
      (is (empty? (:chunks result))))))

(deftest test-extract-text-only
  (testing "lightweight extraction"
    (let [text (extractor/extract-text-only sample-html)]
      (is (clojure.string/includes? text "Active inference"))
      (is (not (clojure.string/includes? text "tracking"))))))

(deftest test-boilerplate-removal
  (testing "removes navigation, footer, scripts"
    (let [html "<html><body>
                  <nav>Navigation</nav>
                  <main><p>Main content here.</p></main>
                  <footer>Footer stuff</footer>
                  <script>alert('x')</script>
                </body></html>"
          result (extractor/extract-content html "https://example.com")
          text (:text result)]
      (is (clojure.string/includes? text "Main content"))
      (is (not (clojure.string/includes? text "Navigation")))
      (is (not (clojure.string/includes? text "Footer stuff"))))))

(deftest test-link-filtering
  (testing "filters out image and binary links"
    (let [html "<html><body>
                  <a href='https://example.com/page'>Page</a>
                  <a href='https://example.com/image.jpg'>Image</a>
                  <a href='https://example.com/doc.pdf'>PDF</a>
                </body></html>"
          result (extractor/extract-content html "https://example.com")
          urls (set (map :url (:links result)))]
      (is (contains? urls "https://example.com/page"))
      (is (not (contains? urls "https://example.com/image.jpg")))
      (is (not (contains? urls "https://example.com/doc.pdf"))))))
