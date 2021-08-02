from collections import defaultdict


class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight




def dijsktra(graph, initial, end,flag=0):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        if flag==0:
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        elif flag==1:
            current_node = max(next_destinations, key=lambda k: next_destinations[k][1])
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

def dijsktra_second_path(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        # print(len(sorted(next_destinations, key=lambda k: next_destinations[k][1])))
        current_node = max(next_destinations, key=lambda k: next_destinations[k][1])
        # print(current_node)
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

# import time
# t1=time.time()
# graph = Graph()
# edges=[('AMES', 'STAR', 1), ('BOIS', 'INL', 1), ('PNNL', 'PNWG', 1), ('BOIS', 'PNWG', 1), ('EQX-ASH', 'EQX-CHI', 1),
#        ('LIGO', 'PNWG', 1), ('BOST', 'PSFC', 1), ('KANS', 'KCNSC', 1), ('ATLA', 'ORAU', 1), ('BOST', 'LNS', 1),
#        ('AMST', 'BOST', 1), ('NERSC', 'SUNN', 1), ('JLAB', 'WASH', 1), ('ATLA', 'SRS', 1), ('GA', 'SUNN', 1), ('HOUS', 'PANTEX', 1),
#        ('EQX-ASH', 'NETL-PGH', 1), ('FNAL', 'STAR', 1), ('LBNL', 'NPS', 1), ('HOUS', 'KANS', 1), ('ATLA', 'ETTP', 1),
#        ('CHIC', 'KANS', 1), ('ALBQ', 'DENV', 1), ('JGI', 'SACR', 1), ('LSVN', 'SUNN', 1), ('LBNL', 'SUNN', 1), ('ALBQ', 'KCNSC-NM', 1),
#        ('CHIC', 'STAR', 1), ('DENV', 'LSVN', 1), ('DENV', 'NREL', 1), ('ATLA', 'Y12', 1), ('DENV', 'KANS', 1), ('DENV', 'NGA-SW', 1),
#        ('LSVN', 'NNSS', 1), ('LLNL', 'SUNN', 1), ('HOUS', 'NASH', 1), ('PNWG', 'SACR', 1), ('CHIC', 'WASH', 1), ('LOND', 'NEWY', 1),
#        ('CERN-513', 'CERN-773', 1), ('ATLA', 'NASH', 1), ('AMST', 'CERN-513', 1), ('CERN', 'CERN-513', 1), ('ANL', 'STAR', 1),
#        ('PPPL', 'WASH', 1), ('SLAC', 'SUNN', 1), ('ATLA', 'ORNL', 1), ('BOST', 'STAR', 1), ('ALBQ', 'LANL', 1), ('NASH', 'WASH', 1),
#        ('EQX-ASH', 'WASH', 1), ('AMST', 'LOND', 1), ('AOFA', 'STAR', 1), ('AOFA', 'WASH', 1), ('ELPA', 'HOUS', 1), ('ELPA', 'SUNN', 1),
#        ('ALBQ', 'SNLA', 1), ('SACR', 'SUNN', 1), ('CERN-773', 'LOND', 1), ('CHIC', 'NASH', 1), ('CERN-513', 'WASH', 1), ('DENV', 'PNWG', 1),
#        ('EQX-ASH', 'NETL-MGN', 1), ('AOFA', 'LOND', 1), ('BNL', 'NEWY', 1), ('CHIC', 'EQX-CHI', 1), ('ATLA', 'WASH', 1), ('BOIS', 'DENV', 1),
#        ('AOFA', 'NEWY', 1), ('BOST', 'NEWY', 1), ('ALBQ', 'ELPA', 1), ('DENV', 'SACR', 1), ('SACR', 'SNLL', 1)]
#
#
# for edge in edges:
#     graph.add_edge(*edge)
#
# print(dijsktra(graph, 'LIGO', 'CERN'))
# print(time.time()-t1,"seconds")
# print(dijsktra_second_path(graph, 'LIGO', 'CERN'))