import os
import numpy as np
import pandas as pd
import json
import argparse
import time
import gc
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc, roc_auc_score
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.preprocessing import LabelEncoder


def one_hot(y):
    
    encoded = np.zeros((len(y), max(y)+1)) #len(np.unique(y))))
    
    for i, label_i in enumerate(y):
        encoded[i, label_i] = 1
            
    return encoded

def get_labels(nodes_with_labels):
    all_labels = set() 
    for idx, row in nodes_with_labels.iterrows():
        node = row['node']
        row_labels = [row['label']]
        row_labels = row["label"].split("=")
        row_labels = row_labels[1:-1]   
        if len(row_labels) == 0:
            continue
        for rl in row_labels:
            all_labels.add(rl)
    all_labels.add('none')
    all_labels.add('nan')
    return list(all_labels)

# generate a dataset with labels from the trained embeddings
def get_X_y(vectors, nodes_with_labels, pos, universe_size, emb_size):
    '''
        Create a train instance X with corresponding labels y.
    '''
    d = universe_size*emb_size
    X = lil_matrix((len(vectors), universe_size))
    y = []
    
    nodes_labels = {str(node): label for node, label in zip(nodes_with_labels['node'], \
                                                            nodes_with_labels['label'])}
    universe = {}
    row_idx = 0
    for node, embedding in vectors.items():
        for i, coord in enumerate(embedding):
            val = coord[pos]
            if val not in universe:
                universe[val] = len(universe)
            col_idx = universe[val]
            X[row_idx, col_idx] = 1
        y.append(nodes_labels[node])
        row_idx += 1
    return X, y


def overlap_kernel(x, y):
    cnt = 0
    for xk, yk in zip(x, y):
        cnt += xk==yk
    return cnt

def gram_matrix(X1, X2, kernel_function):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        if i % 100 == 0:
            print(i)
        xi = X1.iloc[i]
        for j in range(X2.shape[0]):
            xj = X2.iloc[j]
            gram_matrix[i, j] = kernel_function(xi, xj)
    return gram_matrix

def gram_matrix_sparse(X1, X2):
    """(Pre)calculates Gram Matrix K"""

    X2_tr = csr_matrix(X2).transpose()
    gram_matrix = csr_matrix(X1)*X2_tr

    return gram_matrix.todense()

def svm_precomputed_kernel(X, y):
    accs = []
    baccs = []
    aucs_micro = []
    aucs_macro = []
    prec=3
    
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
        gram_train = gram_matrix_sparse(X_train, X_train)
        gram_test = gram_matrix_sparse(X_test, X_train)
        svm_model = svm.SVC(C = 1, kernel="precomputed")
        svm_model.fit(gram_train, y_train)
        y_pred = svm_model.predict(gram_test)
        acc_score =  acc(y_test, y_pred)
        bacc_score = bacc(y_test, y_pred)
        accs.append(acc_score)
        baccs.append(bacc_score)

        cal_model = svm.SVC(C = 1, kernel="precomputed", probability=True)
        cal_model.fit(gram_train, y_train)
        y_pred_prob = cal_model.predict_proba(gram_test)
        y_onehot = one_hot(y_test)
        auc_micro = roc_auc_score(y_onehot, y_pred_prob, average='micro')
        auc_macro = roc_auc_score(y_onehot, y_pred_prob, average='macro')
        aucs_micro.append(auc_micro)
        aucs_macro.append(auc_macro)
        print("Acc: {}, Balanced Acc: {}, AUC-micro: {}, AUC-macro: {}".format(np.round(acc_score, prec), \
                        np.round(bacc_score, prec), np.round(auc_micro, prec), np.round(auc_macro, prec)))
        
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
    universe_size = 100000

    data_dir = os.path.expanduser("../Graphs/"+graphname)
    print(data_dir)

    hop = depth
    nodes_with_labels = pd.read_csv(data_dir + '/data/nodes_with_labels.csv')

    if method == 'random_walk':    
        rwalk_path = data_dir + "/vectors/vectors_rwalk_all_" + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(rwalk_path, "r") as read_file:
            rwalk_vectors = json.load(read_file)
        print('Start RWalk')
        X_rw, y_rw = get_X_y(rwalk_vectors, nodes_with_labels, pos=2, universe_size=universe_size, emb_size=emb_size)
        start = time.time()
        svm_precomputed_kernel(X_rw, y_rw)
        print('Elapsed time', time.time()-start)
        del X_rw
        del y_rw
        gc.collect()
            
    if method == 'nodesketch':         
        nodesketch_path = data_dir + "/vectors/vectors_nodesketch_all_" + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(nodesketch_path, "r") as read_file:
            ns_vectors = json.load(read_file)
        print('Start NS')
        X_ns, y_ns = get_X_y(ns_vectors, nodes_with_labels, pos=2, universe_size=universe_size, emb_size=emb_size)
        start = time.time()
        svm_precomputed_kernel(X_ns, y_ns)
        print('Elapsed time', time.time()-start)
        del X_ns
        del y_ns
        gc.collect()
            
    if method == 'L0':        
        minwise_path = data_dir + "/vectors/vectors_minwise_all_" + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(minwise_path, "r") as read_file:
            l0_vectors = json.load(read_file)
        print('Start L0')
        X_l0, y_l0 = get_X_y(l0_vectors, nodes_with_labels, pos=2, universe_size=universe_size, emb_size=emb_size)
        start = time.time()
        svm_precomputed_kernel(X_l0, y_l0)
        print('Elapsed time', time.time()-start)
        del X_l0
        del y_l0
        gc.collect()
        
        
    if method == 'L1':
        l1_path = data_dir + "/vectors/vectors_l1_all_" + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(l1_path, "r") as read_file:
            l1_vectors = json.load(read_file)
        print('Start L1')
        X_l1, y_l1 = get_X_y(l1_vectors, nodes_with_labels, pos=2, universe_size=universe_size, emb_size=emb_size)
        start = time.time()
        svm_precomputed_kernel(X_l1, y_l1)
        print('Elapsed time', time.time()-start)
        del X_l1
        del y_l1
        gc.collect()
            
    if method == 'L2':        
        l2_path = data_dir + "/vectors/vectors_l2_all_" + str(emb_size) + "_hop_" + str(hop) + ".json"
        with open(l2_path, "r") as read_file:
            l2_vectors = json.load(read_file)
        print('Start L2')
        X_l2, y_l2 = get_X_y(l2_vectors, nodes_with_labels, pos=2, universe_size=universe_size, emb_size=emb_size)
        start = time.time()
        svm_precomputed_kernel(X_l2, y_l2)
        print('Elapsed time', time.time()-start)
        del X_l2
        del y_l2
        gc.collect()
            