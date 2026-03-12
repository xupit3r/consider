(ns consider.web.test-helpers
  (:require [consider.web.crawler :as crawler]
            [hato.client :as hato]))

(def mock-pages
  {"https://en.wikipedia.org/wiki/Active_Inference"
   {:status 200
    :body "<html><head><title>Active Inference</title></head>
           <body><article>
             <p>Active inference is a theory developed by Karl Friston at UCL.
                It is based on the free energy principle.</p>
             <p>Key concepts include variational inference and Markov blankets.</p>
             <a href='https://en.wikipedia.org/wiki/Free_energy_principle'>Free Energy Principle</a>
             <a href='https://en.wikipedia.org/wiki/Karl_Friston'>Karl Friston</a>
             <a href='https://en.wikipedia.org/wiki/Variational_Bayesian_methods'>Variational methods</a>
           </article></body></html>"
    :content-type "text/html"}
   
   "https://en.wikipedia.org/wiki/Free_energy_principle"
   {:status 200
    :body "<html><body><h1>Free Energy Principle</h1><p>A unified theory of brain function.</p></body></html>"
    :content-type "text/html"}
   
   "https://en.wikipedia.org/wiki/Karl_Friston"
   {:status 200
    :body "<html><body><h1>Karl Friston</h1><p>A British neuroscientist at UCL.</p></body></html>"
    :content-type "text/html"}
   
   "https://en.wikipedia.org/w/index.php?search=Active+Inference&title=Special:Search"
   {:status 200
    :body "<html><body>
           <div class='mw-search-result'><a href='https://en.wikipedia.org/wiki/Active_Inference'>Active Inference</a>
           <div class='mw-search-result-text'>A theory of brain function.</div></div>
           </body></html>"
    :content-type "text/html"}

   "https://en.wikipedia.org/robots.txt"
   {:status 200
    :body "User-agent: *\nDisallow: /w/\nAllow: /wiki/"
    :content-type "text/plain"}})

(defn with-mock-http [pages test-fn]
  (with-redefs [hato.client/get (fn [url opts]
                                 (let [page (get pages url)]
                                   (if page
                                     {:status (:status page)
                                      :body (:body page)
                                      :headers {"content-type" (:content-type page)}}
                                     {:status 404 :body "Not Found"})))
                crawler/fetch-robots-txt (fn [domain config]
                                           {:rules {:default {:disallow [] :allow ["/"]}}
                                            :fetched-at (System/currentTimeMillis)})]
    (test-fn)))
