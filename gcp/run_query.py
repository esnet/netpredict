import requests



def run_query(query): # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.get("https://my.es.net/graphql?", params={'query':q})
    print(request)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

        
# The GraphQL query (with a few aditional bits included) itself defined as a multi-line string.       
q = """
{
    mapTopology(name: "routed_toplevel") {
        edges {
            name
            netbeamTraffic
        }
    }
}
"""

result = run_query(q) # Execute the query
print(result)