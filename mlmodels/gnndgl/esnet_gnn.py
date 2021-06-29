import yaml
import os
import sys
import networkx as nx
import dgl

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))


YAMLtopologyfile="../datasets/esnet.yaml"

def read_yaml():
    topo_file=YAMLtopologyfile

    with open(topo_file,'r') as f:
        topo_desc=yaml.load(f)
    
    if topo_desc is None:
        raise ValueError('Error: Load topology from {}'.format(topo_file))
    
    print(topo_desc['links'])
    
def main():
    read_yaml()


main()