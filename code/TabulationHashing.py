from __future__ import division
import copy
from BinaryStream import BinaryStream
from bisect import bisect
import math
import sys
#from sys import maxint
#import numpy as np

#the class implements tabulation hashing. An 32- or 64-bit integers is viewed as c consecutive bit strings. Usually c=4 or c=8
class TabulationHashing:
    
    def __init__(self, filename, rows, shift, bits=64, number = 1):
        if bits != 32 and bits != 64:
            raise Exception('Bits must be either 32 or 64.')
        self.file = open(filename, 'rb')
        self.stream = BinaryStream(self.file)
        self.rows = rows #the number of different hash tables, e..g. 4
        self.cols = 2**shift #the number of different values in each hashtable, e.g. 2^8
        self.shift = shift
        self.bits = bits
        self.number = number
        if number == 1: 
            self.createRandomMatrix()
        else:
            self.createRandomMatrices()
        self.file.close()
        self.tree = None   
        self.trees = None     
        
    def getRndInt(self):
        if self.bits == 32:
            return self.stream.readUInt32()
        else:
            return self.stream.readUInt64()
    
    #initialize tables with random values
    def createRandomMatrix(self):  
        self.rndMatrix = [[0 for j in range(self.cols)] for i in range(self.rows)]#matrix(np.zeros((self.rows, self.cols)));      
        for i in range(self.rows):
            for j in range(self.cols):
                self.rndMatrix[i][j] = self.getRndInt()
    
    def createRandomMatrices(self):
        self.matrices = [[[0 for j in range(self.cols)] for i in range(self.rows)] for k in range(self.number)]
        for k in range(self.number):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrices[k][i][j] = self.getRndInt()
        
                
    def hashValueInt(self, v):
        mask = 2**self.shift - 1 #shift bits set to 1
        hash_value = 0
        i = 0
        while i < self.rows:
            h = v & mask 
            hash_value = hash_value^self.rndMatrix[i][h] #xor the hash value with the random value from the i-th matrix
            i += 1
            v = v >> self.shift
        return hash_value
     
    #assuming random values are 32 bit integers, return a value in [0,1]           
    def hashValue(self, v): 
        mask = 2**self.shift - 1; #(int)Math.pow(2, shift)-1;
        hash_value = 0
        i = 0
        while i < self.rows:
            h = v & mask
            hash_value = hash_value^self.rndMatrix[i][h]
            i += 1
            v = v >> self.shift
        return hash_value/(2**self.bits -1)
    
    #assuming random values are 32 bit integers, return a value in [0,1]           
    def hashValueFromMatrix(self, v, k): 
        mask = 2**self.shift - 1; #(int)Math.pow(2, shift)-1;
        hash_value = 0
        i = 0
        while i < self.rows:
            h = v & mask
            hash_value = hash_value^self.matrices[k][i][h]
            i += 1
            v = v >> self.shift
        return hash_value/(2**self.bits -1)
    
    #get the xor of all but the last table. Not that the "last" table is in the 0-th array.           
    def hashValuePrefix(self, v): 
        mask = 2**self.shift - 1; 
        hash_value = 0
        i = 1
        v = v >> self.shift
        while i < self.rows:
            h = v & mask
            hash_value = hash_value^self.rndMatrix[i][h]
            i += 1
            v = v >> self.shift
        return hash_value
    
    #get the xor of all but the last table           
    def hashValuePrefixFromMatrix(self, v, k): 
        mask = 2**self.shift - 1; 
        hash_value = 0
        i = 1
        v = v >> self.shift
        while i < self.rows:
            h = v & mask
            hash_value = hash_value^self.matrices[k][i][h]
            i += 1
            v = v >> self.shift
        return hash_value
    
    #just for testing
    def sign(self, v):
        if self.hashValue(v) >= 0.5:
            return 1
        else:
            return -1
        
    def getTree(self):
        if self.tree == None:
            raise ValueError('No tree')
        return self.tree
    
    #get the number of different hash tables
    def getNumber(self):
        return self.number
                
    def printMatrix(self):
        for i in range(self.rows):
            print('row ', i)
            for j in range(self.cols):
                print(self.rndMatrix[i][j]/(2**32 -1))
            print('\n')
            
    def hashValuesAtLevel(self):
        table = self.rndMatrix[0]
        values = {}
        slash = 'x' #level 0 containing all values
        for (cnt, val) in enumerate(table):
            bit_repr = ['0' for _ in range(self.bits)]
            missing = self.bits - len(bin(val)) + 2
            bit_repr[missing:] = list(bin(val)[2:])
            vals = []
            if slash in values:
                vals = values[slash]
            vals.append((cnt, ''.join(bit_repr)))
            values[slash] = vals
            for i in range(len(bit_repr)):
                bitstring = ''.join(bit_repr[:i+1])
                hashes = []
                if bitstring in values:
                    hashes = values[bitstring]
                hashes.append((cnt, ''.join(bit_repr)))
                values[bitstring] = hashes
        for k in values.keys():
            values[k] = sorted(values[k])
        return values    
        
    #build a binary search tree that will support efficient search queries
    def preprocessMatrix(self):
        table = self.rndMatrix[0]
        tree_minvalues = {}#[(2**self.cols +1) for _ in range(2**self.bits + 1)] #the minimum indices in each subtree
        for (cnt, val) in enumerate(table):
            bit_repr = ['0' for _ in range(self.bits)]
            missing = self.bits - len(bin(val)) + 2
            bit_repr[missing:] = list(bin(val)[2:])
            for i in range(len(bit_repr)):
                bitstring = ''.join(bit_repr[:i+1])
                if (bitstring) not in tree_minvalues:
                    tree_minvalues[bitstring] = cnt
        self.tree = tree_minvalues
        return tree_minvalues
    
    #build a binary search tree that will support efficient search queries
    def preprocessMatrices(self):
        trees = [{} for _ in range(self.number)]
        for k in range(self.number):
            table = self.matrices[k][self.rows-1]
            tree_minvalues = {}#[(2**self.cols +1) for _ in range(2**self.bits + 1)] #the minimum indices in each subtree
            for (cnt, val) in enumerate(table):
                bit_repr = ['0' for _ in range(self.bits)]
                missing = self.bits - len(bin(val)) + 2
                bit_repr[missing:] = list(bin(val)[2:])
                for i in range(len(bit_repr)):
                    bitstring = ''.join(bit_repr[:i+1])
                    if (bitstring) not in tree_minvalues:
                        tree_minvalues[bitstring] = cnt
            trees[k] = tree_minvalues
        self.trees = trees
        return trees
    
    #find the minimum hash value that is below the limit
    def findMinimum(self, query, limit):
        if query > 2**self.bits:
            raise ValueError('Value must be at most 2** ' + str(self.bits))
        ql = list(bin(query))[2:]
        leading_zeros = ['0' for _ in range(self.bits - len(ql))]
        leading_zeros.extend(ql)
        query_bin = ''.join(leading_zeros)
        i = 1
        minval = ''#['' for _ in range(len(query))]
        while i <= len(query_bin):
            minval += query_bin[i-1]
            if minval[:i] not in self.tree or self.tree[minval[:i]] > limit:
                xored = (int(minval[i-1]) + 1)%2
                lmv = list(minval)
                lmv[i-1] = str(xored)
                minval = ''.join(lmv)
            i += 1
        return (query^int(minval, 2))/(2**self.bits - 1)
    
    #the same as above but the tree is explicitly given
    def findMinimumInLocalTree(self, query, tree, limit):
        if limit < 1:
            raise ValueError('Weight must be at least 1, it is  ' + str(limit))
        if query > 2**self.bits:
            raise ValueError('Value must be at most 2** ' + str(self.bits))
        ql = list(bin(query))[2:]
        leading_zeros = ['0' for _ in range(self.bits - len(ql))]
        leading_zeros.extend(ql)
        query_bin = ''.join(leading_zeros)
        i = 1
        minval = ''
        while i <= len(query_bin):
            minval += query_bin[i-1]
            #print('minval', minval)
            if minval not in tree or tree[minval] > limit:
                xored = (int(minval[i-1]) + 1)%2
                lmv = list(minval)
                lmv[i-1] = str(xored)
                minval = ''.join(lmv)
            i += 1
        return (minval, (query^int(minval, 2))/(2**self.bits-1))
    
    #for the case where we have many hash functions
    def findMinimumInTree(self, query, limit, k):
        if limit < 1:
            raise ValueError('Weight must be at least 1, it is  ' + str(limit))
        if query > 2**self.bits:
            raise ValueError('Value must be at most 2** ' + str(self.bits))
        tree = self.trees[k]
        ql = list(bin(query))[2:]
        leading_zeros = ['0' for _ in range(self.bits - len(ql))]
        leading_zeros.extend(ql)
        query_bin = ''.join(leading_zeros)
        i = 1
        minval = ''
        while i <= len(query_bin):
            minval += query_bin[i-1]
            if minval[:i] not in tree or tree[minval[:i]] > limit:
                xored = (int(minval[i-1]) + 1)%2
                lmv = list(minval)
                lmv[i-1] = str(xored)
                minval = ''.join(lmv)
            i += 1
        return (query^int(minval, 2))/(2**self.bits-1)
    
    def findMinimumValues(self, query, limit, threshold, values):
        if threshold > 0.5:
            threshold = 0.5
        t = 1.0/2**math.floor(abs(math.log(threshold, 2)))
        if limit < 1:
            raise ValueError('Weight must be at least 1, it is  ' + str(limit))
        if query > 2**self.bits:
            raise ValueError('Value must be at most 2** ' + str(self.bits))
        ql = list(bin(query))[2:]
        leading_zeros = ['0' for _ in range(self.bits - len(ql))]
        leading_zeros.extend(ql)
        query_bin = ''.join(leading_zeros)
        #print(query_bin)
        i = 1
        r = 1.0
        while r > t and i <= len(query_bin):
            r = r/2
            i += 1
        minval = query_bin[:i-1]    
        if minval not in values:
            return []
        hashes = values[minval]
        pos = bisect(hashes, (limit, ''))
        res = []
        for j in range(pos):
            v = (query^int(hashes[j][1], 2))/(2**self.bits-1)
            if v <= threshold:
                res.append(v)
        return res
            
        
        
    #xoring the last bit in a given bit string
    def xorValue(self, val):
        l = len(val) - 1
        last_bit = val[l]
        xored = (int(last_bit) + 1) % 2
        lmv = list(val)
        lmv[l] = str(xored)
        return ''.join(lmv)
    
    #delete an entry from the tree
    def deleteFromTree(self, tree, minval):
        if minval not in tree:
            raise ValueError('No such value ' + minval)
        unique = True
        l = len(minval) - 1
        cnt = -1
        while minval in tree and unique:
            #print('l', l, minval)
            #print('deleted', minval, tree[minval])
            del tree[minval]
            xored_minval = self.xorValue(minval)
            if xored_minval in tree:
                unique = False
                cnt = tree[xored_minval]
            else:
                minval = minval[:l]
                l = l-1
        if cnt > -1:
            while l >= 1:
                minval = minval[:l]
                if minval not in tree:
                    raise ValueError('Something is wrong')
                tree[minval] = cnt
                xored = self.xorValue(minval)
                if xored in tree:
                    cnt = min(cnt, tree[xored])
                l = l-1       

    def printTree(self, tree):
        print('Tree')
        for k in tree.keys():
            print(k, tree[k]) 
        print()       
    
if __name__ == "__main__": 
    shift = 12
    tab = TabulationHashing('random/bits.01', 4, shift)
    #print(tab.createRandomMatrix())
    values = tab.hashValuesAtLevel()
    l = 0
    for k in values.keys():
        l += len(values[k])
    print(math.log(l, 2) - shift)
    #tab.printMatrix()
    #tab.printMatrix()
    #print(tab.findMinimumValues(1, 10000, 0.012, values))
    print(sorted(tab.findMinimumValues(1214, 1000, 0.05, values)))
    print(sorted(tab.findMinimumValues(1001, 1000, 0.05, values)))
    sys.exit()
    tree = copy.copy(tab.preprocessMatrix())
    value = 2**14
    #printTree(tree)
    s = 9
    query = tab.hashValuePrefix(value)
    minval = tab.findMinimumInLocalTree(query, tree, s)
    print(minval)
    print(tab.hashValue(2**14))
    print(tab.hashValue(2**14 + 1))
    print(tab.hashValue(2**14 + 2))
    print(tab.hashValue(2**14 + 3))
    print(tab.hashValue(2**14 + 4))
    print(tab.hashValue(2**14 + 5))
    print(tab.hashValue(2**14 + 6))
    print(tab.hashValue(2**14 + 7))
    print(tab.hashValue(2**14 + 8))
    print(tab.hashValue(2**14 + 9))
    #tab.deleteFromTree(tree, minval[0])
    #printTree(tree)
   

    