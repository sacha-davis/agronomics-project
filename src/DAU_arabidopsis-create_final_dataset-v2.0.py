# explanation for the functionality of this file can be found in the associated git issue 

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import time
import random


def fetch_args():
    parser = argparse.ArgumentParser(description="Dataset Creation")

    parser.add_argument('-ds', '--data_path_starr', type=str, default="data/raw/athal_starr_hidra.tsv", help='Path to STARR data')
    parser.add_argument('-di', '--data_path_istarr', type=str, default="data/raw/athal_istarr_hidra.tsv", help='Path to iSTARR data')

    parser.add_argument('-on', '--out_name', type=str, default="data/processed/arabidopsis", help='Path to in data')

    parser.add_argument('-sa', '--sample', type=int, default=1, help='Sample every n rows')

    parser.add_argument('-se', '--separate', type=int, default=0, help='True if train/test/val files should be separate')

    parser.add_argument('-sc', '--starr_control_threshold', type=int, default=30, help='Raw control coverage threshold')
    parser.add_argument('-st', '--starr_treatment_threshold', type=int, default=10, help='Raw treatment coverage threshold')

    parser.add_argument('-ic', '--istarr_control_threshold', type=int, default=30, help='Raw control coverage threshold')
    parser.add_argument('-it', '--istarr_treatment_threshold', type=int, default=5, help='Raw treatment coverage threshold')

    args = parser.parse_args()

    return args


def main():
    args = fetch_args()  # get args

    df = pd.concat([pd.read_csv(args.data_path_starr, sep="\t", header=None), 
                   pd.read_csv(args.data_path_istarr, sep="\t", header=None)],
                   axis=1)

    column_names = ["chromosome", 
                    "start_coord", 
                    "end_coord", 
                    "sequence", 
                    "raw_control_coverage_s", 
                    "raw_treatment_coverage_s", 
                    "norm_control_coverage_s", 
                    "norm_treatment_coverage_s",
                    "chromosome_i", 
                    "start_coord_i", 
                    "end_coord_i", 
                    "sequence_i", 
                    "raw_control_coverage_i", 
                    "raw_treatment_coverage_i", 
                    "norm_control_coverage_i", 
                    "norm_treatment_coverage_i"]
    df.columns = column_names

    print("here!")

    # keep rows with "Chr" in the chromosome column
    df = df[df.chromosome.isin(["Chr"+str(i) for i in range(1,6)])]  

    # select every nth row
    df = df.iloc[::args.sample, :] 

    # get rid of rows with non-standard characters
    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
    df = df[np.logical_not(mask)]  # keep only rows without weird characters 

    # get rid of rows with raw_control_coverage < x and raw_treatment_coverage < y
    df = df[(df.raw_control_coverage_s >= args.starr_control_threshold) & (df.raw_treatment_coverage_s >= args.starr_treatment_threshold)]
    df = df[(df.raw_control_coverage_i >= args.istarr_control_threshold) & (df.raw_treatment_coverage_i >= args.istarr_treatment_threshold)]

    # create target column
    df["target_starr"] = np.log2(df.norm_control_coverage_s/df.norm_treatment_coverage_s)
    df["target_istarr"] = np.log2(df.norm_control_coverage_i/df.norm_treatment_coverage_i)

    df = df[["chromosome", "start_coord", "end_coord", "sequence", "target_starr", "target_istarr"]]

    # assign sets
    df["set"] = "train"
    picked = ["Chr2","Chr4"]
    random.Random(1202).shuffle(picked)
      # set val and test by chromosome we want
    df.loc[df.chromosome == picked[0], "set"] = "val"
    df.loc[df.chromosome == picked[1], "set"] = "test"

    out = str(args.out_name)+"_every_"+str(args.sample)  # format name

    if args.separate == 1:  # if we want train/test/val in different files for batch generator
      df_train = df[df.set == "train"]
      df_test = df[df.set == "test"]
      df_val = df[df.set == "val"]

      df_train.to_csv(out+"_train.csv", index=False)  # write to file
      df_test.to_csv(out+"_test.csv", index=False)  # write to file
      df_val.to_csv(out+"_val.csv", index=False)  # write to file
    else:
      df.to_csv(out+".csv", index=False)


main()
