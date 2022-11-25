# LoNe sampler 

The directory contains code for the AAAI 2023 paper Lone Sampler: Graph node embeddings by coordinated local neighborhood sampling.

## Prerequisites

1. **Python packages** 
The code should work with a standard Python3 version. In addition, one needs some widely used libraries like pandas, numpy, networkx, scikit-learn, etc. The file requirements.txt gives a list of the package versions we used.

2. **Graphs** The provided package contains the Cora graph ([https://graphsandnetworks.com/the-cora-dataset/](URL)) and instructions how to preprocess it. The Citeseer graph ([http://networkrepository.com/citeseer.php](URL)) can be preprocessed using the same code. For the other 4 graphs used in the experimental evaluation, we refer to the respective notebooks. They contain instructions where to download and unpack the data. 

3. **Random numbers**
For tabulation hashing we use a library with preprocessed random numbers that can be downloaded from [https://github.com/jeffThompson/DiehardCDROM/tree/master/CD-ROM
](URL) For larger graphs, one needs to bundle several of the random files together, for example using on a Unix based OS
    

    cat bits.01 bits.02 bits.03 > merged
The file merged should be then stored in the folder named *random*. (The submitted package contains such a file.)


## Running the code

1. **Preprocess a graph**
    
    python3 preprocess.py [--graphname < name >]
The default name is Cora.

2. **Train embeddings**
Run 

    `python3 generate_embeddings.py` 

Usage:

    generate_embeddings.py [-h] [--graphname [GRAPHNAME]]
                                  [--embedding_size EMBEDDING_SIZE]
                                  [--depth DEPTH] [--method [METHOD]]
                                  [--sketchsize SKETCHSIZE]

    
The default arguments are Cora, embedding size 5, depth 2, default method L2, and sketchsize 1. (See the paper for details.)

3. **Train a linear model**

    `python3 linear_svm_training.py` 

    Train and evaluate a linear SVM model for node classification for a given graph and a given method for embedding generation. The parameters are the as above, just type the option `--help` for instructions.

4. (Optional) *Train a kernel model*

    For comparison, one can also train a kernel SVM model

## Note 
This version only supports a single core embedding training. Please contact kutzkov_at_gmail.com for the working code for the multicore version.

