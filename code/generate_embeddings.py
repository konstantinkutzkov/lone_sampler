import os
import sys
import networkx as nx
import pandas as pd
import numpy as np
import argparse
import json
import random
import math
import copy
import time
import gc


def read_graph_from_edge_list(data_dir, filename, nodedata):
    G = nx.Graph()
    path = data_dir + "/data/" + filename
    cnt = 0
    with open(path, 'r') as edgefile: # os.path.join(data_dir, filename),
        for line in edgefile:
            cnt += 1
            line_split = line.split(':')
            if len(line_split) > 1:
                l0 = line_split[0]
                l1 = line_split[1]
                u = l0.strip()
                v = l1.strip()
                if u in nodedata and v in nodedata:
                    G.add_edge(u, v)  
    return G

# generate a random value in (min_val, 1]
def get_rnd_value(rand_gen, min_val):
    if min_val >=1:
        raise Exception("Minimum must be less than 1")
    rval = 0
    while rval < min_val:
        #r = random.randint(0, 1e14)
        rval = random.random() #rand_gen.hashValue(r)
    return rval

# generate a random int in [start, end]
def get_rnd_int_in_range(start, end):
    return random.randint(start, end)

# a standard random walk starting from a node for 'depth' hops 
def random_walk(G, node, depth, features):# , rand_gen):
    node = str(node)
    cnt = 0
    curr_node = node
    while cnt < depth and G.degree[curr_node] > 0:
        nbrs = [curr_node] + [nbr for nbr in G.neighbors(curr_node)]
        curr_node = nbrs[get_rnd_int_in_range(0, len(nbrs)-1)]
        cnt += 1
    subject, features_node = features[curr_node]
    # return a random feature describing the node
    if len(features_node.values())==0:
        print(features[curr_node])
    random.seed(73)
    w = random.choices(population=list(features_node.keys()), weights=list(features_node.values()), k=1)[0]
    return curr_node, subject, w


# for each node generate a number of samples, i.e. the embedding size, by random walks
def all_nodes_random_walk(G, depth, nr_walks, features): #, rand_gen):
    vectors = {}
    for node in G.nodes():
        vectors[node] = [None for _ in range(nr_walks)]
        for walk in range(nr_walks):
            sample, subject, feature = random_walk(G, node, depth, features) #, rand_gen)
            vectors[node][walk] = (sample, subject, feature)
    return vectors


def generate_dicts_for_features(filename, nr_dicts):
    dicts = [{} for _ in range(nr_dicts)]
    f = open(filename, 'r')
    random.seed(42)
    for line in f:
        feature = line.strip()
        for d in dicts:
            d[feature] = np.random.random() 
    f.close()
    return dicts


def update_dict(d, k, min_val, seed):
    random.seed(seed)
    if k not in d:
        d[k] = get_rnd_value(min_val)


# the sampling procedure used in NodeSketch
def ioffe_sampling(arr, weights):
    min_val = 1e6
    node_sample = None
    feature_sample = None
    weight_sample = 0
    label_sample = None
    for node, vals in arr.items():
        feature, weight, label = vals[0], vals[1], vals[2]
        node_rnd_val = -math.log(weights[node])/weight
        if node_rnd_val < min_val:
            min_val = node_rnd_val
            node_sample = node
            feature_sample = feature
            weight_sample = weight
            label_sample = label
    return node_sample, feature_sample, weight_sample, label_sample

def update_arr(arr, new_node):
    if new_node[0] in arr:
        arr[new_node[0]] = (new_node[1], arr[new_node[0]][1] + new_node[2], new_node[3])# [1] += new_node[2]
    else:
        arr[new_node[0]] = (new_node[1], new_node[2], new_node[3])
    return arr


# the NodeSketch algorithm
def nodesketch_iter(G, nodedata, depth, emb_size, feature_dicts): #, rand_gen):
    
    node_labels = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        node_labels_i = node_labels[i]
        feats_rnd_i = feature_dicts[i]
        for node, feats in nodedata.items():
            arr = {}
            for f, weight_f in feats[1].items():
                arr[f] = (f, weight_f, feats[0])
                #arr.append((f, f, weight_f, feats[0]))
            _, feature_sample, weight_sample, label_sample = ioffe_sampling(arr, feats_rnd_i)
            node_labels_i[node] = (node, feature_sample, weight_sample, label_sample)
            
    print('Sampled features')
    
    node_rnd_vals_all = [{} for _ in range(emb_size)]
    for t in range(emb_size):
        random.seed(1223*t)
        for u in G.nodes():
            node_rnd_vals_all[t][u] = random.random()
            
    node_labels_all = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    node_labels_all[0] = node_labels
    for d in range(depth):
        node_labels_iter = node_labels_all[d]
        print('Iteration', d)
        random.seed(31*d)
        # node_rnd_vals = {}
        
        for t in range(emb_size):
            node_labels_iter_t = node_labels_iter[t]
            for u in G.nodes():
                node_sample_u, feature_sample_u, weight_sample_u, label_u = node_labels_iter_t[u]
                arr_u = {node_sample_u: (feature_sample_u, weight_sample_u, label_u)} 
                #[(node_sample_u, feature_sample_u, weight_sample_u, label_u)]
                for v in G.neighbors(u):
                    node_sample_v, feature_sample_v, weight_sample_v, label_v = node_labels_iter_t[v]
                    update_arr(arr_u, (node_sample_v, feature_sample_v, weight_sample_v, label_v))
                node_labels_all[d+1][t][u] = ioffe_sampling(arr_u, node_rnd_vals_all[t]) 
                
    node_embeddings = [{n:[] for n in G.nodes()} for _ in range(depth+1)]
    for d in range(depth+1):
        for u in G.nodes():
            for nl in node_labels_all[d]:
                node_embeddings[d][u].append((nl[u][0], nl[u][3], nl[u][1]))
    return node_embeddings
                    

# initialize random numbers for nodes and features for each embedding 
def init_dicts(nodedata, emb_size, rand_gen): #, rand_gen=rand_gen):
    min_val = 1e-6
    cnt = 0
    feats_rnd = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        if i%10 == 0:
            print('i=', i)
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            for f, weight_f in feats[1].items():
                cnt += 1
                update_dict(feats_rnd_i, f, rand_gen, min_val, seed=17*cnt)
    return feats_rnd

# L0 sampling using minwise hashing
def generate_minwise_samples(G, nodedata, feats_rnd, depth, emb_size):
    node_labels = [{} for _ in range(emb_size)]
    for i in range(emb_size):
        node_labels_i = node_labels[i]
        feats_rnd_i = feats_rnd[i]
        for node, feats in nodedata.items():
            min_feature_value = 1e3
            min_feature = None
            for f in feats[1]:
                if feats_rnd_i[f] < min_feature_value:
                    min_feature = f
                    min_feature_value = feats_rnd_i[f]
            node_labels_i[node] = (min_feature_value, node, feats[0], min_feature)
            
    node_labels_all = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    node_labels_all[0] = node_labels
    for d in range(depth):
        node_labels_iter = node_labels_all[d]
        print('Iteration', d)
        for u in G.nodes():
            for t in range(emb_size):
                w_u = node_labels_iter[t][u]
                for v in G.neighbors(u):
                        w_u = min(node_labels_iter[t][v], w_u)
                node_labels_all[d+1][t][u] = w_u
            
    node_embeddings = [{n:[] for n in G.nodes()} for _ in range(depth+1)]
    for d in range(depth+1):
        for u in G.nodes():
            for nl in node_labels_all[d]:
                node_embeddings[d][u].append((nl[u][1], nl[u][2], nl[u][3]))
    return node_embeddings



# generating Lp samples from the k-hop neighborhood
# top is the summary size of the frequent items mining algorithm
def generate_Lp_samples(G, p, nodedata, feature_dicts, depth, emb_size, top):
    assert p==1 or p==2
    sketches = [[{} for _ in range(emb_size)] for _ in range(depth+1)]
    
    cnt = 0
    # generate the random values for each node (attribute)
    for u, (label, features) in nodedata.items():
        cnt += 1
        if cnt % 3000 == 0:
            
            print('nodes processed', cnt)
        for i in range(emb_size):
            feats_rnd_i = feature_dicts[i] #feats_rnd[i]
            
            triples = []
            for f, w_f in features.items():
                w_rnd = w_f/feats_rnd_i[f]**(1/p)   # get_rnd_value(rand_gen, min_val) 
                triples.append((w_rnd, f, u))
            triples = sorted(triples, reverse=True)
            to_subtract = 0
            if len(triples) > top:
                to_subtract = triples[top][0]
            top_triples = triples[:top]
            sketches[0][i][u]= {tr[1] : (tr[0]-to_subtract, tr[2]) for tr in top_triples}
    
    # iterate over neighborhoods and maintain the heaviest nodes
    for d in range(depth):
        print('ITERATION', d)
        for i in range(emb_size):
            # print('emb', emb)
            sketch_iter_emb = copy.deepcopy(sketches[d][i])
            # print(len(sketch_iter_emb))
            new_sketches = {}
            for u in G.nodes():
                if u not in sketch_iter_emb:
                    continue
                sketch_u = copy.deepcopy(sketch_iter_emb[u])
                for v in G.neighbors(u):
                    sketch_v = sketch_iter_emb[v]
                    for t, (w_f, f) in sketch_v.items():
                        sketch_u.setdefault(t, (0, None))
                        weight = sketch_u[t][0] + w_f
                        sketch_u[t] = (weight, f)
                triples = []
                for node, feat_node in sketch_u.items():
                    triples.append((feat_node[0], feat_node[1], node))
                    
                # mining heavy hitters
                triples = sorted(triples, reverse=True)
                to_subtract = 0
                if len(triples) > top:
                    to_subtract = triples[top][0]
                top_triples = triples[:top]
                
                new_sketches[u] = {tr[2] : (tr[0]-to_subtract, tr[1]) for tr in top_triples}
            sketches[d+1][i] = new_sketches
    return sketches


def get_embeddings_l1_2(nodedata, sketches, emb_size):
    embeddings = [{} for _ in range(len(sketches))]
    for d in range(len(sketches)):
        for node in nodedata.keys():
            embeddings[d][node] = []
    
    for d in range(len(sketches)):
        for e in range(emb_size):
            for node, dct in sketches[d][e].items():
                max_node = None
                max_word = None
                max_weight = -1
                for sampled_node, ww in dct.items(): # ww: weight word
                    if ww[0] > max_weight:
                        max_node = sampled_node
                        max_word = ww[1]
                        max_weight = ww[0]
                label = nodedata[max_word][0]
                if max_weight > 0:
                    embeddings[d][node].append((max_word, label, max_node))
    return embeddings

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Embedding generation for the different methods.")
    parser.add_argument('--graphname', nargs='?', default='Cora', help='Input graph name')
    parser.add_argument('--embedding_size', type=int, default=50, help='The dimensionality of the embeddings')
    parser.add_argument('--depth', type=int, default=2, help='The depth of the node neighborhood')
    parser.add_argument('--method', nargs='?', default='L2', 
              help="The method for embedding generation ['random_walk', 'nodesketch', 'L0', 'L1', 'L2']")
    parser.add_argument('--sketchsize', type=int, default=3, help='The sketch size for L1/L2 sampling')
    
    args = parser.parse_args()
    graphname = args.graphname
    emb_size = args.embedding_size # how many samples per node to generate 
    assert emb_size > 1 
    depth = args.depth # the depth of the local neighborhood
    assert depth > 0
    method = args.method
    sketchsize = args.sketchsize
    assert sketchsize > 0
    if method not in set(['random_walk', 'nodesketch', 'L0', 'L1', 'L2']):
        print('The available methods are ', ['random_walk', 'nodesketch', 'L0', 'L1', 'L2'])
        sys.exit()
    edgename = 'all'
    write_to_file = True
    
    data_dir = os.path.expanduser("../Graphs/"+graphname)

    nodedata_path = data_dir + "/data/nodedata" + ".json" 
    with open(nodedata_path, "r") as read_file:
        nodedata = json.load(read_file)

    print('Reading graph {}\n'.format(graphname))
    G = read_graph_from_edge_list(data_dir, edgename + "_graph_edges.txt", nodedata)

    if method == 'random_walk':
        print('\nRandom walk sampling')
        # random walks on the full graph
        vectors_rwalk = []
        start = time.time()
        for d in range(depth+1):
            vectors_rw = all_nodes_random_walk(G, d, emb_size, features=nodedata) 
            vectors_rwalk.append(vectors_rw)
            end = time.time()
        print('Elapsed time Random Walk', end-start)

 
        for d in range(depth+1):
            jsonpath = data_dir + "/vectors/vectors_rwalk_" + edgename + "_" + \
                str(emb_size) + "_hop_" + str(d) + ".json"
            with open(jsonpath, 'w') as outfile:
                json.dump(vectors_rwalk[d], outfile)
        del vectors_rwalk
        gc.collect()

    # NodeSketch sampling
    if method == 'nodesketch':
        print('\nNodeSketch sampling')
        feature_dicts = generate_dicts_for_features(data_dir + '/data/words_indices.txt', emb_size)
        start = time.time()
        vectors_ns= nodesketch_iter(G, nodedata, depth=depth, emb_size=emb_size, feature_dicts=feature_dicts) 
        end = time.time()
        print('Elapsed time Nodesketch', end-start)

        for d in range(depth+1):
            jsonpath = data_dir + "/vectors/vectors_nodesketch_" + edgename + "_" + \
                    str(emb_size) + "_hop_" + str(d) + ".json"
            with open(jsonpath, 'w') as outfile:
                json.dump(vectors_ns[d], outfile)   
        del vectors_ns
        gc.collect()

    # L0 sampling
    if method == 'L0':
        print('\nL0 sampling')
        feature_dicts = generate_dicts_for_features(data_dir + '/data/words_indices.txt', emb_size)
        start = time.time()
        vectors_mw = generate_minwise_samples(G, nodedata, feature_dicts, depth=depth, emb_size=emb_size)
        end = time.time()
        print('Elapsed time L0', end-start)

        for d in range(depth+1):
            jsonpath = data_dir + "/vectors/vectors_minwise_" + edgename + "_"  + \
                str(emb_size) + "_hop_" + str(d) + ".json"
            with open(jsonpath, 'w') as outfile:
                json.dump(vectors_mw[d], outfile)    
        del vectors_mw
        gc.collect()

    # L1 sampling
    if method == 'L1':
        print('\nL1 sampling')
        top = sketchsize
        feature_dicts = generate_dicts_for_features(data_dir + '/data/words_indices.txt', emb_size)
        start = time.time()
        sketches_l1 = generate_Lp_samples(G, 1, nodedata, feature_dicts, depth=depth, emb_size=emb_size, top=top)
        end = time.time()
        print('Elapsed time L1', end-start)

        vectors_l1 = get_embeddings_l1_2(nodedata, sketches_l1, emb_size)
        del sketches_l1
        gc.collect()

        for d in range(depth+1):
            jsonpath = data_dir + "/vectors/vectors_l1_" + edgename + "_" + str(emb_size) + "_hop_" + str(d) + ".json"
            with open(jsonpath, 'w') as outfile:
                json.dump(vectors_l1[d], outfile)
        del vectors_l1
        gc.collect()

    # L2 sampling
    if method == 'L2':
        print('\nL2 sampling')
        top = sketchsize
        feature_dicts = generate_dicts_for_features(data_dir + '/data/words_indices.txt', emb_size)
        start = time.time()
        sketches_l2 = generate_Lp_samples(G, 2, nodedata, feature_dicts, depth=depth, emb_size=emb_size, top=top)
        end = time.time()
        print('Elapsed time L2', end-start)
        vectors_l2 = get_embeddings_l1_2(nodedata, sketches_l2, emb_size)
        for d in range(depth+1):
            jsonpath = data_dir + "/vectors/vectors_l2_" + edgename + "_" + str(emb_size) + "_hop_" + str(d) + ".json"
            with open(jsonpath, 'w') as outfile:
                json.dump(vectors_l2[d], outfile)
        del sketches_l2
        del vectors_l2
        gc.collect()