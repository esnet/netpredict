from flask import Flask, render_template, request

from network_map import graph_main, graph_bar, graph_nodes, nodes_join_line, Shortest_path_graph

app = Flask(__name__)
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

graph_type=[
    'Hourly',
    'Weekly',
    'Next 3 Months'
]
# map_data = json.load(open('map-data.json'))
# # print(map_data)
# coordinates_location = []
# for city in map_data["data"]['mapTopology']['nodes']:
#     coordinates_location.append([city['name'], city['x'] * 10 + 150, city['y'] * 9 + 100])
#     # plt.annotate(city['name'],xy=city['name'])
# df = pd.DataFrame(coordinates_location, columns=['Name', 'X', 'Y'])
# # print(df.to_numpy().resize())
# plt.scatter('X','Y',data=df)
# for i in range(df.shape[0]):
#     plt.text(x=df.X[i]+0.5,y=df.Y[i]+0.5 , s=df.Name[i],
#              fontdict=dict(color='red', size = 4))
#                 # bbox = dict(facecolor='yellow', alpha = 0.5))
#
# # # print(plt.imread('static/images/NetPredict-05.png')[:,:,0])
# plt.imshow(plt.imread('static/images/NetPredict-05.png'))
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# # plt.bar([i[0] for i in predictionsData], [i[1] for i in predictionsData])
# # plt.xlabel("Time")
# # plt.ylabel("TimeData Transfer(in TB)")
# # plt.xticks(range(len([i[0] for i in predictionsData])), rotation=75)
# plt.axis('off')
# plt.savefig('static/images/out.png',dpi=720)
# # # plt.show()
# graph=['14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may']
# predict=[0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6]
graph_main()
graph_bar(predictionsData)

Graph_Nodes=graph_nodes()
Graph_node_line=nodes_join_line()
graph,predict,barColors,shortest_path_data=[],[],[],[]
source,destination,data_transmit,gr_type='','','',''
@app.route('/')
def home():
    print(graph,predict,barColors)
    return render_template("index.html",
                           data_source=sites,
                           data_destination=sites.copy(),
                           source=source, destination=destination,
                           data_transmit=data_transmit,
                           data_transfer=dataTransferredOptions,
                           prediction_data=predictionsData,
                           graph_type=graph_type,
                           Graph_Nodes=Graph_Nodes,
                           graph_date=graph,
                           graph_predict=predict,
                           graph_color=barColors,
                           Graph_node_line=Graph_node_line,
                           Shortest_Path=[])

@app.route('/home',methods=['GET','POST'])
def updated_home():
    global graph
    global predict
    global barColors
    global shortest_path_data
    global source
    global destination
    global data_transmit
    global gr_type
    global data_transfer
    if request.method=='POST':
        print('*****************',request.form)
        source,destination,data_transfer,gr_type = request.form['source'],request.form['destination'],\
                                                request.form['data_transfer'],request.form['data_type']

        barColors = ["#f00", "#0f0", "blue", "orange",
                     "brown", "yellow", "purple", "pink",
                     "black","violet","gray","silver",]
        if gr_type=='Hourly':
            graph = ['01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00','09:00', '10:00', '11:00', '12:00',]
            predict = [0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6, 0.33, 0.66, 0.95, 0.6]
        elif gr_type=='Weekly':
            graph = ['14 may', '15 may', '16 may', '17 may', '18 may', '19 may', '20 may', '21 may']
            predict = [0.11, 0.13, 0.07, 0.04, 0.33, 0.66, 0.95, 0.6]
            # barColors = ["red", "green", "blue", "orange", "brown", "yellow", "purple", "pink"]
        elif gr_type=='Next':
            graph = ['July', 'August', 'September']
            predict = [0.11, 0.13, 0.12]
            # barColors = ["red", "green", "blue"]
            # print('------------------inside')
        else:
            graph = []
            predict = []
            barColors = []

        shortest_path_data = Shortest_path_graph(source, destination)
        # print("----------", shortest_path_data)
        # print("**********",list(set(Graph_node_line)-set(shortest_path_data)))
        return render_template("index.html",
               data_source=sites,
               data_destination=sites.copy(),
               source=source, destination=destination,
               data_transmit=data_transmit,
               gr_type=gr_type,
               data_transfer=dataTransferredOptions,
               prediction_data=predictionsData,
               graph_type=graph_type,
               graph_date=graph,
               graph_predict=predict,
               graph_color=barColors[:len(predict)],
               Graph_node_line=Graph_node_line,
               Shortest_Path=shortest_path_data,
               Graph_Nodes=Graph_Nodes,
               path_color='blue'
               )


@app.route('/optimize_home/<source>/<destination>/<traffic>')
def optimize_path(source,destination,traffic):

    if float(traffic)>=0.5:
        color='red'
    else:
        color='blue'
    print('*****************', request.form)
    return render_template("index.html",
                           data_source=sites,
                           data_destination=sites.copy(),
                           source=source, destination=destination,
                           data_transmit=data_transmit,
                           gr_type=gr_type,
                           data_transfer=dataTransferredOptions,
                           prediction_data=predictionsData,
                           graph_type=graph_type,
                           Graph_Nodes=Graph_Nodes,
                           graph_date=graph,
                           graph_predict=predict,
                           graph_color=barColors,
                           Graph_node_line=Graph_node_line,
                           Shortest_Path=Shortest_path_graph(source, destination),
                           path_color=color)


if __name__ == "__main__":
    app.run(debug=True)
