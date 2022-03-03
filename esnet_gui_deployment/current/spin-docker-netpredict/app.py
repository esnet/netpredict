import os
import mysql.connector
from flask import Flask, render_template, request, g, url_for 

# from database_connectivity import read_database
from network_map import graph_main, graph_bar, graph_nodes, nodes_join_line, Shortest_path_graph

config = {
    "database": {
        "user": "netpred",
        "password": os.environ.get("MYSQL_PASSWORD"),
        "host": "netpredictdb",
        "database": "netpredictdb",
    }, 
} 

# config = {
#     "database": {
#         "user": "netpred",
#         "password": "rootroot",
#         "host": "localhost",
#         "database": "netpredictdb",
#     }, 
# } 

app = Flask(__name__)

@app.before_first_request
def before_first_request():
    connection = create_connection(**config["database"]) 
    try:
        cursor = connection.cursor()
        cursor.execute('create table predicted_values(duration VARCHAR(50), predict DOUBLE);')
        graph = ['14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may']
        predict = [0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6]
        for (duration,predicted) in zip(graph,predict):
            cursor.execute(f'insert into predicted_values (duration,predict) values ("{duration}",{predicted});')
            connection.commit()
    except:
        pass  
    else:
        cursor.close()
    connection.close()
    
@app.before_request
def before_request():
    g.connection = create_connection(**config["database"])
    g.cursor = g.connection.cursor()
    
@app.after_request
def after_request(response):
    g.cursor.close()
    g.connection.close()
    return response

def create_connection(host=None, user=None, password=None, database=None):
    return mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database)
    

def read_database():
  g.cursor.execute("select name, filename from data")
  
  graph, predict = [i[0] for i in g.cursor], [i[1] for i in g.cursor]

  return [graph,predict]
    

sites = [
    'ALBQ',
    'AMES',
    'AMST',
    'ANL',
    'AOFA',
    'ATLA',
    'BNL',
    'BOIS',
    'BOST',
    'CERN',
    'CERN513',
    'CERN773',
    'CHIC',
    'DENV',
    'ELPA',
    'EQX-ASH',
    'EQX-CHI',
    'ETTP',
    'FNAL',
    'GA',
    'HOUS',
    'INL',
    'JGI',
    'JLAB',
    'KANS',
    'KCNSC',
    'KCNSC-NM',
    'LANL',
    'LBNL',
    'LIGO',
    'LLNL',
    'LNS',
    'LOND',
    'LSVN',
    'NASH',
    'NERSC',
    'NETL-MGN',
    'NETL-PGH',
    'NEWY',
    'NGA-SW',
    'NNSS',
    'NPS',
    'NREL',
    'ORAU',
    'ORNL',
    'PANTEX',
    'PNNL',
    'PNWG',
    'PPPL',
    'PSFC',
    'SACR',
    'SLAC',
    'SNLA',
    'SNLL',
    'SRS',
    'STAR',
    'SUNN',
    'WASH',
    'Y12',
]
predictionsData = [
    ['14 may', 0.01, ['FNAL', 'STAR']],
    ['15 may', 0.13, ['NREL', 'DENV', 'KANS']],
    ['16 may', 0.07, ['SACR', 'DENV']],
    ['17 may', 0.04, ['FNAL', 'STAR']],
    ['18 may', 0.33, ['NREL', 'DENV', 'KANS']],
    ['19 may', 0.66, ['NREL', 'DENV', 'KANS']],
    ['20 may', 0.95, ['FNAL', 'STAR']],
    ['21 may', 0.6, ['SACR', 'DENV']],
]
# predictionsData = [
#     ['2017-01-24T00:00', 0.01, ['FNAL', 'STAR']],
#     ['2017-01-24T01:00', 0.13, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T02:00', 0.07, ['SACR', 'DENV']],
#     ['2017-01-24T03:00', 0.04, ['FNAL', 'STAR']],
#     ['2017-01-24T04:00', 0.33, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T05:00', 0, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T06:00', 0, ['FNAL', 'STAR']],
#     ['2017-01-24T07:00', 0, ['SACR', 'DENV']],
#     ['2017-01-24T08:00', 0.95, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T09:00', 1.12, ['FNAL', 'STAR']],
#     ['2017-01-24T10:00', 0.66, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T11:00', 0.06, ['SACR', 'DENV']],
#     ['2017-01-24T12:00', 0.3, ['FNAL', 'STAR']],
#     ['2017-01-24T13:00', 0.05, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T14:00', 0.5, ['SACR', 'DENV']],
#     ['2017-01-24T15:00', 0.24, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T16:00', 0.02, ['SACR', 'DENV']],
#     ['2017-01-24T17:00', 0.98, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T18:00', 0.46, ['SACR', 'DENV']],
#     ['2017-01-24T19:00', 0.8, ['FNAL', 'STAR']],
#     ['2017-01-24T20:00', 0.39, ['SACR', 'DENV']],
#     ['2017-01-24T21:00', 0.4, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T22:00', 0.39, ['NREL', 'DENV', 'KANS']],
#     ['2017-01-24T23:00', 0.28, ['SACR', 'DENV']],
# ]
dataTransferredOptions = [
    'Less than 1GB',
    '1GB -  10GB',
    '10GB - 20GB',
    '20GB - 40GB',
    '40GB - 60GB',
    '60GB - 80GB',
    '80GB - 100GB',
    '100GB - 120GB'
]

graph_type = [
    'Hourly',
    'Weekly',
    'Monthly'
]

graph_main()
graph_bar(predictionsData)

Graph_Nodes = graph_nodes()
Graph_node_line = nodes_join_line()
graph, predict, shortest_path_data = [], [], []
source, destination, data_transmit, gr_type, xlabel = '', '', '', '', ''
graph_predict_data = []


@app.route('/test')
def test():
    graph, predict=read_database()
    # graph = ['14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may']
    # predict = [0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6]
    graph_predict_data = []
    [graph_predict_data.append({"x": int(i.split()[0]), "y": j}) for (i, j) in zip(graph, predict)]
    return render_template("COLUMN_TEST.html",
                           Graph_Nodes=Graph_Nodes,
                           Graph_node_line=Graph_node_line,
                           Shortest_Path=[],
                           column_graph=graph_predict_data,
                           xlabel="Time (Houly, Weekly, Monthly)"
                           )


@app.route('/')
def home():
    global city
    global x_coordinates
    global y_coordinates
    graph = ['1 may', '2 may', '3 may', '4 may', '5 may', '6 may', '7 may', '8 may', '9 may', '10 may', '11 may', '12 may', '13 may', '14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may', '22 may', '23 may', '24 may','25 may', '26 may', '27 may', '28 may', '29 may', '30 may', '31 may', '32 may', '33 may', '34 may', '35 may', '36 may', '37 may', '38 may', '39 may', '40 may', '41 may', '42 may', '43 may', '44 may', '45 may', '46 may', '47 may', '48 may']
    predict = [0.1, 0.2, 0.07, 0.04, 0.33, 0.36, 0.45, 0.6, 0.11, 0.13, 0.07, 0.1, 0.2, 0.5, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6, 0.11, 0.13, 0.07, 0.8, 0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6, 0.33, 0.66, 0.95, 0.6, 0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6, 0.33, 0.66, 0.95, 0.6,]
    graph_predict_data = []
    [graph_predict_data.append({"x": int(i.split()[0]), "y": j}) for (i, j) in zip(graph, predict)]

    return render_template("index.html",
                           column_graph=graph_predict_data,
                           xlabel="Time (Houly, Weekly, Monthly)",
                           data_source=sites,
                           data_destination=sites.copy(),
                           source=source, destination=destination,
                           data_transmit=data_transmit,
                           data_transfer=dataTransferredOptions,
                           prediction_data=predictionsData,
                           graph_type=graph_type,
                           zip_graph=zip(Graph_Nodes, Graph_node_line),
                           zip_short=zip(Graph_Nodes, shortest_path_data))


@app.route('/home', methods=['GET', 'POST'])
def updated_home():
    global shortest_path_data
    global source
    global destination
    global data_transmit
    global gr_type
    global data_transfer
    global graph_predict_data
    global xlabel
    if request.method == 'POST':
        request_data = request.form
        source, destination, data_transfer, gr_type = request_data['source'], request_data['destination'], \
                                                      request_data['data_transfer'], request_data['data_type']

        if gr_type == 'Hourly':
            graph = ['01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00','09:00', '10:00', '11:00', '12:00','13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00','21:00', '22:00', '23:00', '24:00','25:00', '26:00', '27:00', '28:00', '29:00', '30:00', '31:00', '32:00','33:00', '34:00', '35:00', '36:00','37:00', '38:00', '39:00', '40:00', '41:00', '42:00', '43:00', '44:00','45:00', '46:00', '47:00', '48:00']
            predict = [0.08, 0.13, 0.07, 0.04, 0.33, 0.66, 0.40, 0.6, 0.33, 0.66, 0.25, 0.4, 0.09, 0.13, 0.07, 0.04, 0.33, 0.66, 0.25, 0.6, 0.33, 0.66, 0.05, 0.6,0.11, 0.13, 0.07, 0.04, 0.33, 0.16, 0.5, 0.6, 0.33, 0.26, 0.45, 0.6, 0.11, 0.13, 0.07, 0.04, 0.33, 0.6, 0.95, 0.2, 0.33, 0.46, 0.35, 0.2]
            graph_predict_data = []
            [graph_predict_data.append({"x": int(i.split(':')[0]), "y": j}) for (i, j) in zip(graph, predict)]
            xlabel = "Hourly"
        elif gr_type == 'Weekly':
            # graph = ['14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may']
            # predict = [0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6]
            graph, predict = read_database()
            graph_predict_data = []
            [graph_predict_data.append({"x": int(i.split()[0]), "y": j}) for (i, j) in zip(graph, predict)]
            xlabel = "Weekly"
            #xlabel = graph[0].split()[1]
        elif gr_type == 'Next':
            graph = [(7,'July',2021), (8,'August',2021), (9,'September',2021)]
            predict = [0.11, 0.13, 0.12]
            graph_predict_data=[]
            [graph_predict_data.append({"x": int(i[0]), "y": j}) for (i, j) in zip(graph, predict)]
            xlabel = graph[0][2]
        else:
            graph_predict_data = []

        shortest_path_data = Shortest_path_graph(source, destination, Graph_Nodes)
        return render_template("index.html",
                               column_graph=graph_predict_data,
                               xlabel=xlabel,
                               data_source=sites,
                               data_destination=sites.copy(),
                               source=source, destination=destination,
                               data_transmit=data_transmit,
                               gr_type=gr_type,
                               data_transfer=dataTransferredOptions,
                               prediction_data=predictionsData,
                               graph_type=graph_type,
                               zip_graph=zip(Graph_Nodes,Graph_node_line),
                               zip_short=zip(Graph_Nodes,shortest_path_data),
                               path_color='navy')



@app.route('/optimize_home/<source>/<destination>/<traffic>')
def optimize_path(source, destination, traffic):
    global xlabel
    # print('*****************', request.form)
    return render_template("index.html",
                           column_graph=graph_predict_data,
                           xlabel=xlabel,
                           data_source=sites,
                           data_destination=sites.copy(),
                           source=source, destination=destination,
                           data_transmit=data_transmit,
                           gr_type=gr_type,
                           data_transfer=dataTransferredOptions,
                           prediction_data=predictionsData,
                           graph_type=graph_type,
                           zip_graph=zip(Graph_Nodes, Graph_node_line),
                           zip_short=zip(Graph_Nodes, Shortest_path_graph(source, destination, Graph_Nodes, traffic)),
                           path_color="red" if float(traffic) >= 0.5 else "navy")



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0') 
