#!/usr/bin/env python

"""
Adapted from https://github.com/zygmuntz/phraug/blob/master/csv2libsvm.py

Convert train_numeric.csv into libSVM format.
The file train_numeric.csv is composed of 970 columns.
The first line of the file describes the titles of the columns.
The first column is an Id and is ignored for the output file.
It might be relevant to let it as a feature though.
The last column (Response) contains the labels.

Usage:
    python csv2libsvm.py /path/to/train_numeric.csv /path/to/train_numeric.svm.txt

    Adjust LINE_START and LINE_END to select a subset of the dataset.

"""

import sys
import csv
from collections import defaultdict

# PARAMETERS
LABEL_INDEX = 969 # column Response
LINE_START = 0
LINE_END = 100000

def construct_line( label, line ):
    new_line = []
    if float( label ) == 0.0:
        label = "0"
    new_line.append( label )

    for i, item in enumerate( line ):
        if item == '' or float( item ) == 0.0:
            continue
        new_item = "%s:%s" % ( i , item )
        new_line.append( new_item )
    new_line = " ".join( new_line )
    new_line += "\n"
    return new_line

input_file = sys.argv[1]
output_file = sys.argv[2]

i = open( input_file, 'r' )
o = open( output_file, 'w' )

reader = csv.reader( i )

# skip headers
headers = next(reader)

line_count = 0
for line in reader:

    line_count += 1

    if line_count <= LINE_START:
        continue

    if line_count >= LINE_END:
        break

    if label_index == -1:
        label = '1'
    else:
        label = line.pop( label_index )

    # skip Id column
    line.pop(0)

    new_line = construct_line( label, line )
    o.write( new_line )

    if line_count%10000 == 0:
        print('%d lines processed.' % line_count)
