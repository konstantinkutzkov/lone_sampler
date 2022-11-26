import os
import argparse
import networkx as nx
import pandas as pd
import numpy as np
import json
import sys
from BinaryStream import BinaryStream

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Preprocess citing networks.")
    parser.add_argument('--graphname', nargs='?', default='Cora',
                            help='Input graph name')
    args = parser.parse_args()
    graphname = args.graphname
    print("Preprocessing ", graphname)
    if graphname != 'Cora' and graphname != 'Citeseer':
        print('The preprocessing routine supports only Cora and Citeseer.')
        sys.exit()

    data_dir = os.path.expanduser("../Graphs/" + graphname)

    if not os.path.isdir(data_dir):
        print('The given path does not exist. The graph must be located in the folder ../Graphs/')
        sys.exit()

    print("Reading graph data")
    edges_df = pd.read_csv(os.path.join(data_dir, graphname.lower() + ".cites"), 
                            sep='\t', header=None, names=["target", "source"])

    content = pd.read_csv(os.path.join(data_dir, graphname.lower() +".content"), \
                            sep='\t', header=None)

    feature_names = ["w-{}".format(ii) for ii in range(content.shape[1]-2)]
    column_names =  ['node'] + feature_names + ["label"]
    nodedata_df = pd.read_csv(os.path.join(data_dir, graphname.lower() +".content"), \
                            sep='\t',  names=column_names)

    print("Collecting data per node")
    nodedata = {}
    for idx, row in nodedata_df.iterrows():
        nodedata[str(row['node'])] = ('label=' + row['label'], {})
        for c in nodedata_df.columns:
            if c[0] == 'w':
                if row[c] != 0:
                    nodedata[str(row['node'])][1][c] =1


    ndata_df = pd.DataFrame()
    rows = []
    for node, feats in nodedata.items():
        row = {}
        row['node'] = node
        row['label'] = feats[0]
        rows.append(row)
    ndata_df = pd.DataFrame(rows)    
    ndata_df.to_csv(data_dir + '/data/nodes_with_labels.csv', index=False)    

    jsonpath = data_dir + "/data/nodedata.json" 
    print(jsonpath)
    with open(jsonpath, 'w') as outfile:
        json.dump(nodedata, outfile)


    nodes = set()
    labels = set()
    word_indices = set()
    for node, features in nodedata.items():
        nodes.add(node)
        labels.add(features[0])
        for w in features[1]:
            word_indices.add(str(w))


    print("Writing graph data.")    
    nodes_path = data_dir + "/data/graph_nodes.txt"
    with open(nodes_path, 'w') as outfile:
        for node in nodes:
            outfile.write(str(node) + '\n')
            
    labels_path = data_dir + "/data/labels.txt" 
    with open(labels_path, 'w') as outfile:
        for label in labels:
            outfile.write(label + '\n')
            
    words_path = data_dir + "/data/words_indices.txt" 
    with open(words_path, 'w') as outfile:
        for wi in word_indices:
            outfile.write(wi + '\n')
            
    edges = []
    for idx, row in edges_df.iterrows():
        edges.append((str(row['target']).strip(), str(row['source']).strip()))

    G = nx.Graph()
    for edge in edges:
        u = edge[0]
        v = edge[1]
        if u in nodedata and v in nodedata:
            G.add_edge(u, v)

    edges_path = data_dir + "/data/all_graph_edges.txt" 
    with open(edges_path, 'w') as outfile:
        for edge in G.edges():
            outfile.write(edge[0] + ':' + edge[1] + '\n')