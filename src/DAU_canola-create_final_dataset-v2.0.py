# explanation for the functionality of this file can be found in the associated git issue 

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def fetch_args():
    parser = argparse.ArgumentParser(description="Dataset Creation")

    parser.add_argument('-da', '--data_path', type=str, default="data/raw/hidra.tsv", help='Path to in data')
    parser.add_argument('-nc', '--num_chunks', type=int, default=70, help='Number of chunks used to create train/test/val split')
    parser.add_argument('-co', '--cut_outliers', type=bool, default=False, help='Whether to get rid of outliers or not')

    args = parser.parse_args()

    return args


def trim(chunks):  # takes in list of dfs, returns list but with no overlaps
    chunks[0] = chunks[0][:-14]  # gets rid of overlap in first chunk
    for i in range(1, len(chunks)-1):  # gets rid of overlap in middle chunks
        chunks[i] = chunks[i][14:-14]
    chunks[-1] = chunks[-1][14:]  # gets rid of overlap in last chunk
    return chunks


def print_contents(train, val, test):
    # see target GC stats
    print("GC % mean in train set:", np.mean(train["avg"]))
    print("GC % std dev of train set:", np.std(train["avg"]), "\n")
    print("GC % mean of val set:", np.mean(val["avg"]))
    print("GC % std dev of val set:", np.std(val["avg"]), "\n")
    print("GC % mean of test set:", np.mean(test["avg"]))
    print("GC % std dev of test set:", np.std(test["avg"]), "\n\n")
    # see target summary stats
    print("Target mean of train set:", np.mean(train[8]))
    print("Target std dev of train set:", np.std(train[8]), "\n")
    print("Target mean of val set:", np.mean(val[8]))
    print("Target std dev of val set:", np.std(val[8]), "\n")
    print("Target mean of test set:", np.mean(test[8]))
    print("Target std dev of test set:", np.std(test[8]), "\n\n")


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_path, sep="\t", header=None)

    # count number of each NT in a sequence
    for nt in ["A", "T", "C", "G"]:
        df[nt] = df[3].str.count(nt)
    df["CG"] = df["C"] + df["G"]
    df["avg"] = df["CG"]/145

    df_chloro = df[df[0] == "NC_016734.1"]  # restrict to only chloroplast data

    chloro_chunks = trim(np.array_split(df_chloro, args.num_chunks))  # return chunks with overlapping sequences removed

    # create train/val/test lists
    training = []
    validation = []
    test = []
    for i in range(len(chloro_chunks)):   # divides each of n chunks into train/test/val, append to train/test/val lists
      idx = int((chloro_chunks[i].shape[0] - 29*2)*0.1) + 14  # index of 10% mark after trimming
        
      trimmed = trim([chloro_chunks[i][:idx], chloro_chunks[i][idx:-idx], chloro_chunks[i][-idx:]])  # get rid of all overlapping sequences between train/test/val
        
      test.append(trimmed[0])
      training.append(trimmed[1])
      validation.append(trimmed[2])

    # set "set" variable to indicate set allegiance
    train = pd.concat(training)
    train["set"] = "train"
    val = pd.concat(validation)
    val["set"] = "val"
    test = pd.concat(test)
    test["set"] = "test"

    # remove outliers from test set and set out file path
    if args.cut_outliers: 
      train['zscore'] = stats.zscore(train[8])
      train = train[(train.zscore < 2) & (train.zscore > -2)]
      train = train.drop(columns=["zscore"])
      out_file_name = "data/processed/hidra_chloroplast_"+str(args.num_chunks)+"_dropout.csv"
    else:
      out_file_name = "data/processed/hidra_chloroplast_"+str(args.num_chunks)+".csv"

    print_contents(train, val, test)

    # create final dataset
    final = pd.concat([train, val, test])
    final = final.sort_index()
    final.columns = ["organelle", 
                    "start_coords", 
                    "end_coords", 
                    "sequence", 
                    "control_raw_coverage", 
                    "treatment_raw_coverage",
                    "control_norm_coverage",
                    "treatment_norm_coverage",
                    "target", 
                    "A", 
                    "T", 
                    "C", 
                    "G", 
                    "CG", 
                    "avg",
                    "set"]
    final = final.drop(columns=["A","T","C","G","CG","avg"])
    print("Final dataset shape:", final.shape)
                    
    final.to_csv(out_file_name, index=False)  # write to file

main()
