# coding: utf-8
from __future__ import division

import struct
import sys

#FILE_NAME = "GoogleNews-vectors-negative300.bin"
FILE_NAME = "E:\\TUDelft\\ThirdQuarter\\information retrival\\GoogleNews-vectors-negative300.bin"
MAX_VECTORS = 200000  # This script takes a lot of RAM (>2GB for 200K vectors), if you want to use the full 3M embeddings then you probably need to insert the vectors into some kind of database
FLOAT_SIZE = 4  # 32bit float

vectors = dict()

with open(FILE_NAME, 'rb') as f:
    c = None
    count2 = 0
    # read the header
    header = ""
    while c != "\n":
        c = f.read(1)
        header += c
        count2  = count2 + 1
        print(count2)
        if(count2 == 1000000):
            break;
        #print("the header is",header)
        #print("The while loop is finished")
    total_num_vectors, vector_len = (int(x) for x in header.split())
    num_vectors = min(MAX_VECTORS, total_num_vectors)

    print
    "Number of vectors: %d/%d" % (num_vectors, total_num_vectors)
    print
    "Vector size: %d" % vector_len
    count = 0
    while len(vectors) < num_vectors:

        word = ""
        while True:
            c = f.read(1)
            if c == " ":
                break
            word += c
            count = count + 1
            print (count)
            if(count == 1000000):
                break
        binary_vector = f.read(FLOAT_SIZE * vector_len)
        vectors[word] = [struct.unpack_from('f', binary_vector, i)[0]
                         for i in xrange(0, len(binary_vector), FLOAT_SIZE)]

        sys.stdout.write("%d%%\r" % (len(vectors) / num_vectors * 100))
        sys.stdout.flush()

import cPickle

print
"\nSaving..."
with open(FILE_NAME[:-3] + "pcl", 'wb') as f:
    cPickle.dump(vectors, f, cPickle.HIGHEST_PROTOCOL)