import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import networkx as nx
from matplotlib import pyplot as plt

from shortest_path import Graph, dijsktra   #, dijsktra_second_path

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

def graph_bar(data):
    date,value=[i[0] for i in data],[i[1] for i in data]
    # print(date,value)
    plt.bar(date,value)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel("Date")
    plt.ylabel("Data Transfer")
    plt.savefig('static/images/bar_graph.png', dpi=720, bbox_inches='tight')
    plt.close()
    # plt.show()

def add_edge_to_graph(G, e1, e2, w):
    G.add_edge(e1, e2, weight=w)

def graph_main():
    map_data = json.load(open('map-data.json'))
    coordinates_location = []
    for city in map_data["data"]['mapTopology']['nodes']:
        coordinates_location.append([city['name'], city['x'] * 10 + 150, city['y'] * 9 + 100])
    # print(coordinates_location)
    df = pd.DataFrame(coordinates_location, columns=['Name', 'X', 'Y'])
    # print(df)
    names=LabelEncoder()
    label_transform=names.fit_transform(df['Name'])
    names_classes={j:i for i,j in zip(label_transform,names.classes_)}
    # print(names_classes)
    edges=[]
    for edge in map_data["data"]['mapTopology']['edges']:
        # for i in x:
        a,b=edge['name'].split('--')
        edges.append((names_classes[a],names_classes[b],1))
    # print(edges)
    G = nx.Graph()
    points = [(i,1200-j) for (c,i,j) in coordinates_location]#[(1, 10), (8, 10), (10, 8), (7, 4), (3, 1)]  # (x,y) points
    # edges = [(0, 1, 10), (1, 2, 5), (2, 3, 25), (0, 3, 3), (3, 4, 8)]  # (v1,v2, weight)

    for i in range(len(edges)):
        add_edge_to_graph(G, points[edges[i][0]], points[edges[i][1]], edges[i][2])

    # you want your own layout
    # pos = nx.spring_layout(G)
    pos = {point: point for point in points}
    # print(pos)
    # add axis
    fig, ax = plt.subplots()
    for i in range(df.shape[0]):
        plt.text(x=df.X[i]+0.5,y=1200-df.Y[i]+0.5 , s=df.Name[i],
                 fontdict=dict(color='red', size = 8))
    # plt.imshow(plt.imread('static/images/NetPredict-05.png'))
    nx.draw(G, pos=pos, node_color='k', ax=ax)
    nx.draw(G, pos=pos, node_size=2, ax=ax)  # draw nodes and edges
    # nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names
    # draw edge weights
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    plt.axis("off")
    ax.set_xlim(0, 2700)
    ax.set_ylim(0, 1200)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.savefig('static/images/out.png', dpi=720,bbox_inches='tight')
    plt.close()
    plt.show()

def graph_nodes():
    map_data = json.load(open('map-data.json'))
    coordinates_location = []
    for city in map_data["data"]['mapTopology']['nodes']:
        coordinates_location.append([city['name'], int(city['x']), int(city['y'])])
    # # print(coordinates_location)
    df = pd.DataFrame(coordinates_location, columns=['Name', 'X', 'Y'])
    return df.to_numpy().tolist()

def nodes_join_line():
    map_data = json.load(open('map-data.json'))
    edges = []
    nodes_of_sites=[{i[0]: (i[1], i[2])} for i in graph_nodes()]
    nodes_of_site={}
    [nodes_of_site.update(i) for i in nodes_of_sites]
    # print(nodes_of_site)
    line_coordinates=[]
    for edge in map_data["data"]['mapTopology']['edges']:
        # for i in x:
        a, b = edge['name'].split('--')
        line_coordinates.append((a,nodes_of_site[a],b,nodes_of_site[b]))
        # print(a,b)
    return line_coordinates

def Shortest_path_graph(source,destination,graph_nodes,traffic=0.3):
    # map_data = json.load(open('map-data.json'))
    # edges = []
    try:
        nodes_of_sites=[{i[0]: (i[1], i[2])} for i in graph_nodes]
        nodes_of_site={}
        [nodes_of_site.update(i) for i in nodes_of_sites]
        # print(nodes_of_site)
        Shortest_path_coordinates=[]

        graph = Graph()
        edges = [('AMES', 'STAR', 1), ('BOIS', 'INL', 1), ('PNNL', 'PNWG', 1), ('BOIS', 'PNWG', 1),
                 ('EQX-ASH', 'EQX-CHI', 1), ('LIGO', 'PNWG', 1), ('BOST', 'PSFC', 1), ('KANS', 'KCNSC', 1),
                 ('ATLA', 'ORAU', 1), ('BOST', 'LNS', 1), ('AMST', 'BOST', 1), ('NERSC', 'SUNN', 1), ('JLAB', 'WASH', 1),
                 ('ATLA', 'SRS', 1), ('GA', 'SUNN', 1), ('HOUS', 'PANTEX', 1), ('EQX-ASH', 'NETL-PGH', 1),
                 ('FNAL', 'STAR', 1), ('LBNL', 'NPS', 1), ('HOUS', 'KANS', 1), ('ATLA', 'ETTP', 1), ('CHIC', 'KANS', 1),
                 ('ALBQ', 'DENV', 1), ('JGI', 'SACR', 1), ('LSVN', 'SUNN', 1), ('LBNL', 'SUNN', 1), ('ALBQ', 'KCNSC-NM', 1),
                 ('CHIC', 'STAR', 1), ('DENV', 'LSVN', 1), ('DENV', 'NREL', 1), ('ATLA', 'Y12', 1), ('DENV', 'KANS', 1),
                 ('DENV', 'NGA-SW', 1), ('LSVN', 'NNSS', 1), ('LLNL', 'SUNN', 1), ('HOUS', 'NASH', 1), ('PNWG', 'SACR', 1),
                 ('CHIC', 'WASH', 1), ('LOND', 'NEWY', 1), ('CERN-513', 'CERN-773', 1), ('ATLA', 'NASH', 1),
                 ('AMST', 'CERN-513', 1), ('CERN', 'CERN-513', 1), ('ANL', 'STAR', 1), ('PPPL', 'WASH', 1),
                 ('SLAC', 'SUNN', 1), ('ATLA', 'ORNL', 1), ('BOST', 'STAR', 1), ('ALBQ', 'LANL', 1), ('NASH', 'WASH', 1),
                 ('EQX-ASH', 'WASH', 1), ('AMST', 'LOND', 1), ('AOFA', 'STAR', 1), ('AOFA', 'WASH', 1), ('ELPA', 'HOUS', 1),
                 ('ELPA', 'SUNN', 1), ('ALBQ', 'SNLA', 1), ('SACR', 'SUNN', 1), ('CERN-773', 'LOND', 1),
                 ('CHIC', 'NASH', 1), ('CERN-513', 'WASH', 1), ('DENV', 'PNWG', 1), ('EQX-ASH', 'NETL-MGN', 1),
                 ('AOFA', 'LOND', 1), ('BNL', 'NEWY', 1), ('CHIC', 'EQX-CHI', 1), ('ATLA', 'WASH', 1), ('BOIS', 'DENV', 1),
                 ('AOFA', 'NEWY', 1), ('BOST', 'NEWY', 1), ('ALBQ', 'ELPA', 1), ('DENV', 'SACR', 1), ('SACR', 'SNLL', 1)]

        for edge in edges:
            graph.add_edge(*edge)
        # if float(traffic)>0.5:
        #     path_source_destination=dijsktra_second_path(graph, source,destination)
        # else:
        #     path_source_destination=dijsktra(graph, source,destination)

        path_source_destination=dijsktra(graph, source, destination, 1 if float(traffic)>0.5 else 0)
        # print(path_source_destination)
        for (index,edge) in enumerate(path_source_destination):
            # for i in x:
            try:
                a, b = edge,path_source_destination[index+1]     #edge['name'].split('--')
            except IndexError:
                continue
            Shortest_path_coordinates.append((a,nodes_of_site[a],b,nodes_of_site[b]))
            # # print(a,b)
        return Shortest_path_coordinates
    except:
        return []
# # print(Shortest_path_graph())
# nodes_join_line()
# graph_bar(predictionsData)
# graph_main()
# # print(graph_nodes())