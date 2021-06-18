import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import networkx as nx
from matplotlib import pyplot as plt
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
    print(date,value)
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
    print(coordinates_location)
    df = pd.DataFrame(coordinates_location, columns=['Name', 'X', 'Y'])
    names=LabelEncoder()
    label_transform=names.fit_transform(df['Name'])
    names_classes={j:i for i,j in zip(label_transform,names.classes_)}
    print(names_classes)
    edges=[]
    for edge in map_data["data"]['mapTopology']['edges']:
        # for i in x:
        a,b=edge['name'].split('--')
        edges.append((names_classes[a],names_classes[b],1))
    print(edges)
    G = nx.Graph()
    points = [(i,1200-j) for (c,i,j) in coordinates_location]#[(1, 10), (8, 10), (10, 8), (7, 4), (3, 1)]  # (x,y) points
    # edges = [(0, 1, 10), (1, 2, 5), (2, 3, 25), (0, 3, 3), (3, 4, 8)]  # (v1,v2, weight)

    for i in range(len(edges)):
        add_edge_to_graph(G, points[edges[i][0]], points[edges[i][1]], edges[i][2])

    # you want your own layout
    # pos = nx.spring_layout(G)
    pos = {point: point for point in points}
    print(pos)
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
    # plt.show()

# graph_bar(predictionsData)