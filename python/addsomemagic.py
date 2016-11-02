#!/usr/bin/env python

import sys
import csv
from collections import defaultdict
import pandas

# PARAMETERS
LABEL_INDEX = 969 # column Response
MAGIC_INDEX = 970

for stage in ['train']: #['train', 'test']:
    MAGIC_FILE = '/home/nath/Projects/Kaggle/data/uncompressed-data/{0}_magic.csv'.format(stage)
    INPUT_FILE = '/home/nath/Projects/Kaggle/data/uncompressed-data/{0}_numeric.csv'.format(stage)
    OUTPUT_FILE = '/home/nath/Projects/Kaggle/data/svm-format/{0}_numeric_with_magic.svm'.format(stage)

    # read magic
    magic = pandas.read_csv(MAGIC_FILE).set_index('Id')

    def construct_line( label, line, id_ ):
        new_line = []
        if float( label ) == 0.0:
            label = "0"
        new_line.append( label )

        for i, item in enumerate( line ):
            if item == '' or float( item ) == 0.0:
                continue
            new_item = "%s:%s" % ( i , item )
            new_line.append( new_item )

        # add some magic
        for i in range(4):
            new_line.append("%s:%s" % (MAGIC_INDEX+i, magic.loc[int(id_), '0_¯\_(ツ)_/¯_'+str(i+1)]))

        new_line = " ".join( new_line )
        new_line += "\n"
        return new_line

    i = open( INPUT_FILE, 'r' )
    o = open( OUTPUT_FILE, 'w' )

    reader = csv.reader( i )

    # skip headers
    headers = next(reader)

    line_count = 0
    for line in reader:

        # get label
        if stage=='train':
            label = line.pop(LABEL_INDEX)
        else:
            label = 0

        # Id column
        id_ = line.pop(0)

        new_line = construct_line( label, line, id_ )
        o.write(new_line)

        line_count += 1

        if line_count%10000 == 0:
            print('%d lines processed.' % line_count)
