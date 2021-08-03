#!/usr/bin/env python3

import numpy as np
import pandas as pd
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description="Convert CSV data to HDF5")

    parser.add_argument("-i", "--input", metavar="STR", type=str, default="data.csv", help="Input file (default: data.csv)")
    parser.add_argument("-o", "--output", metavar="STR", type=str, default="data.h5", help="Output file (default: data.h5)")

    args = parser.parse_args()
    
    print("Reading input file...")
    df = pd.read_csv(args.input)

    print(df)
    df=df.iloc[:,1:]

    # Save to HDF5
    print("Creating output file...")
    df.to_hdf(args.output, 'data', mode='w', format='table')
    #del df    # allow df to be garbage collected
    print("HDF5 file created...")

    # Append more data
    #df2 = pd.DataFrame(np.arange(10).reshape((4,2))*10, columns=['C1', 'C2'])
    #df2.to_hdf(filename, 'data', append=True)

    #print(pd.read_hdf(filename, 'data'))
