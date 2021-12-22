from elasticsearch import Elasticsearch
    
es = Elasticsearch(["https://el.gc1.prod.stardust.es.net:9200"], timeout=60)
res = es.search(body={
   "size":0,
   "_source":False,
   "aggs":{
      "ifaces":{
        "terms":{
          "field":"meta.id",
          "size": 25000
        }
      }
   },
   "query":{
      "bool":{
         "filter":[
            {
               "range":{
                  "start":{
                     "gte":"now-15m/m",
                     "lte":"now"
                  }
               }
            },
            {
                "query_string": {
                  "analyze_wildcard": True,
                  "query": "(meta.service_type:\"ies\") OR (!meta.service_type:*)"
                }
            }
         ]
      }
   }
}, index="sd_public_interfaces")

print(res)