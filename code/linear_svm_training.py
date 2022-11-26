import os
import numpy as np
import pandas as pd
import json
import argparse
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from TabulationHashing import TabulationHashing
import gc


def one_hot(y):
    
    encoded = np.zeros((len(y), len(np.unique(y))))
    for i, label_i in enumerate(y):
        encoded[i, label_i] = 1
            
    return encoded

def map_to_binary_vectors(vectors, explicit_vector_size, rand_gen, universe_size, pos=2):
    '''
        Map the discrete embedding vectors to low-dimensional binary vectors using a hash function.
    '''
    binary_vectors = {}
    universe = {}
    for node, embedding in vectors.items():
        vec = [0 for _ in range(explicit_vector_size)]
        for i, coord in enumerate(embedding):
            val = coord[pos]
            if val not in universe:
                universe[val] = len(universe)
            idx = i*universe_size + universe[val]
            hash_value = rand_gen.hashValueInt(idx)%explicit_vector_size
            vec[hash_value] = 1
        binary_vectors[node] = vec
    return binary_vectors #, universe
    
    
# generate a dataset with labels from the trained embeddings
def get_X_y(vectors, nodes_with_labels, x):
    '''
        Create a train instance X with corresponding labels y.
    '''
    X = []
    y = []
    labels = {}
    
    for idx, row in nodes_with_labels.iterrows():
        node = row['node']
        row_label = row['label']
        if row_label != row_label:
            continue
        features = vectors[str(node)]
        X.append(features)
        if row_label not in labels:
            labels[row_label] = len(labels)
        y.append(labels[row_label])
    X = pd.DataFrame(X)
    return X, y


def svm_classification(X, y):      
    
    accs = []
    baccs = []
    aucs_micro = []
    aucs_macro = []
    nr_iters = 10

    C = 1
    prec = 3
    for i in range(nr_iters):    
        print('Iter', i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=i)
        model = LinearSVC(C=C, random_state=0, tol=1e-5, max_iter=3000)
        y_train = np.ravel(y_train)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        
        model = LinearSVC(C=C, random_state=0, tol=1e-5, max_iter=3000)
        model = CalibratedClassifierCV(model)
        model.fit(X_train, y_train) 
        y_pred_prob = model.predict_proba(X_test)
        acc_score =  acc(y_test, y_pred)
        bacc_score = bacc(y_test, y_pred)
        
        y_onehot = one_hot(y_test)
        auc_micro = roc_auc_score(y_onehot, y_pred_prob, average='micro')
        auc_macro = roc_auc_score(y_onehot, y_pred_prob, average='macro')
        
        print("Acc: {}, Balanced Acc: {}, AUC-micro: {}, AUC-macro: {}".format(np.round(acc_score, prec), \
                        np.round(bacc_score, prec), np.round(auc_micro, prec), np.round(auc_macro, prec)))
        accs.append(acc_score)
        baccs.append(bacc_score)
        aucs_micro.append(auc_micro)
        aucs_macro.append(auc_macro)
        
    print('\nMean Accuracy: {}, std: {}'.format(np.round(np.mean(accs), prec), np.round(np.std(accs), prec)))
    print('Mean balanced accuracy: {}, std: {}'.format(np.round(np.mean(baccs), prec), \
                                                       np.round(np.std(baccs), prec)))
    print('Mean AUC-micro: {}, std: {}'.format(np.round(np.mean(aucs_micro), prec), \
                                               np.round(np.std(aucs_micro), prec)))
    print('Mean AUC-macro: {}, std: {}'.format(np.round(np.mean(aucs_macro), prec), \
                                               np.round(np.std(aucs_macro), prec)))
    
    


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Embedding generation for the different methods.")
    parser.add_argument('--graphname', nargs='?', default='Cora', help='Input graph name')
    parser.add_argument('--embedding_size', type=int, default=50, help='The dimensionality of the embeddings')
    parser.add_argument('--depth', type=int, default=2, help='The depth of the node neighborhood')
    parser.add_argument('--method', nargs='?', default='L2', help='The method for embedding generation')

    args = parser.parse_args()
    graphname = args.graphname
    emb_size = args.embedding_size # how many samples per node to generate 
    assert emb_size > 1 
    depth = args.depth # the depth of the local neighborhood
    assert depth > 0
    method = args.method

    labelsamples = False


    data_dir = os.path.expanduser("../Graphs/"+graphname)
    print(data_dir)

    emb_size = 50
    print("\n\nEmbedding size", emb_size)
    
    eps = 0.01
    explicit_vector_size = int(emb_size/eps)
    universe_size=100000
    x=2

    randompath = "../random/merged"
    rand_gen = TabulationHashing(randompath, rows=4, shift=16)
    nodes_with_labels = pd.read_csv(data_dir + '/data/nodes_with_labels.csv')

    hop=depth

    if method == 'random_walk':
        rwalk_path = data_dir + "/vectors/vectors_rwalk_all" + '_' + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(rwalk_path, "r") as read_file:
            rwalk_vectors = json.load(read_file)
        print('Start RWalk based model')
        rwalk_binary = map_to_binary_vectors(rwalk_vectors, explicit_vector_size, rand_gen, universe_size)
        X_rw, y_rw = get_X_y(rwalk_binary, nodes_with_labels, x=x)
        svm_classification(X_rw, y_rw)
        del rwalk_binary
        del X_rw
        del y_rw
        gc.collect()

    if method == 'nodesketch':
        ns_path = data_dir + "/vectors/vectors_nodesketch_all" +  '_' + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(ns_path, "r") as read_file:
            ns_vectors = json.load(read_file)
        print('\nStart NodeSketch')
        ns_binary = map_to_binary_vectors(ns_vectors, explicit_vector_size, rand_gen, universe_size)
        X_ns, y_ns = get_X_y(ns_binary, nodes_with_labels, x=x)
        del ns_binary
        gc.collect()
        start = time.time()
        svm_classification(X_ns, y_ns)
        print('Elapsed time', time.time()-start)
        del X_ns
        del y_ns
        gc.collect()


    if method == 'L0':
        l0_path = data_dir + "/vectors/vectors_minwise_all" + '_' + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(l0_path, "r") as read_file:
            l0_vectors = json.load(read_file)
            print('\nStart L0')
        l0_binary = map_to_binary_vectors(l0_vectors, explicit_vector_size, rand_gen, universe_size)
        X_l0, y_l0 = get_X_y(l0_binary, nodes_with_labels, x=x)
        del l0_binary
        gc.collect()
        svm_classification(X_l0, y_l0)
        del X_l0
        del y_l0
        gc.collect()

    if method == 'L1':
        l1_path = data_dir + "/vectors/vectors_l1_all" + '_' + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(l1_path, "r") as read_file:
            l1_vectors = json.load(read_file)
        print('\nStart L1')
        l1_binary = map_to_binary_vectors(l1_vectors, explicit_vector_size, rand_gen, universe_size)
        X_l1, y_l1 = get_X_y(l1_binary, nodes_with_labels, x=x)
        del l1_binary
        gc.collect()
        svm_classification(X_l1, y_l1)
        del X_l1
        del y_l1
        gc.collect()

    if method == 'L2':
        l2_path = data_dir + "/vectors/vectors_l2_all" + '_' + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(l2_path, "r") as read_file:
            l2_vectors = json.load(read_file)


        print('\nStart L2')
        l2_binary = map_to_binary_vectors(l2_vectors, explicit_vector_size, rand_gen, universe_size)
        X_l2, y_l2 = get_X_y(l2_binary, nodes_with_labels, x=x) 
        start = time.time()
        svm_classification(X_l2, y_l2)
        print('Elapsed time', time.time()-start)
        del l2_binary
        del X_l2
        del y_l2
        gc.collect()