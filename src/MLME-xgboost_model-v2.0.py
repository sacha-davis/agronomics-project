import argparse
import keras
import warnings, logging
import numpy as np
import pandas as pd
import datetime, time, os
import json
import random
import os

import xgboost

from sklearn.metrics import r2_score
from scipy.stats import spearmanr  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

warnings.filterwarnings('ignore')
logging.disable(1000)

nts = ["A", "T", "C", "G"]  # list of single nucleotides


def fetch_args():
    parser = argparse.ArgumentParser(description="Frequency Model")

    parser.add_argument('-da', '--data_path', type=str, default="data/processed/hidra_chloroplast_70.csv", help='Learning Rate')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('-ep', '--num_epochs', type=int, default=100, help='Total Number of Epochs')
    parser.add_argument('-pa', '--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('-ms', '--max_batch_steps', type=int, default=-1, help='Maximum number of steps in a batch, by default goes through entire batch')
    parser.add_argument('-op', '--optimizer', type=str, default='adam', help='Keras optimizer -- pick from [adam]')
    parser.add_argument('-vt', '--verbose_training', type=int, default=1, help='Verbose Training')

    parser.add_argument('-mo', '--include_mononuc_freq', type=int, default=1, help='Use single nucleotide frequencies')
    parser.add_argument('-di', '--include_dinuc_freq', type=int, default=0, help='Use dinucleotide frequencies')
    parser.add_argument('-tr', '--include_trinuc_freq', type=int, default=0, help='Use trinucleotide frequencies')

    parser.add_argument('-st', '--include_starr', type=int, default=0, help='Whether to include the STARR target value in the output vector')
    parser.add_argument('-is', '--include_istarr', type=int, default=1, help='Whether to include the iSTARR target value in the output vector')

    args = parser.parse_args()

    return args


def return_y(args, df):  # based on what to include, returns y array
    if args.include_starr:
      if args.include_istarr:
        y = np.array(pd.concat([df["target_starr"], df["target_istarr"]], axis=1))
      else:
        y = np.array(df["target_starr"].tolist())
    else:
      y = np.array(df["target_istarr"].tolist())

    return y


def train_test_val(args):
    include = []  # captures all sequences we are including as input features
    if args.include_mononuc_freq == 1:  include += nts
    if args.include_dinuc_freq == 1:    include += [nt1+nt2 for nt1 in nts for nt2 in nts]
    if args.include_trinuc_freq == 1:   include += [nt1+nt2+nt3 for nt1 in nts for nt2 in nts for nt3 in nts]

    df = pd.read_csv(args.data_path)

    print("read csv")

    for item in include:  # create new columns with the counts of sequences in "include"
      print("including", item)
      df[item] = df.sequence.str.count(item)

    train_df = df[df.set == "train"]
    X_train = np.array(train_df[include])
    y_train = return_y(args, train_df)

    val_df = df[df.set == "val"]
    X_val = np.array(val_df[include])
    y_val = return_y(args, val_df)

    test_df = df[df.set == "test"]
    X_test = np.array(test_df[include])
    y_test = return_y(args, test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test


def save_results(args, dir_path, X_train, X_test, X_val, y_train, y_test, y_val, saved_model):
    if args.include_starr + args.include_istarr == 2:

      train_predictions = saved_model.predict(X_train)
      val_predictions = saved_model.predict(X_val)
      test_predictions = saved_model.predict(X_test)

      starr_scores = [[str(r2_score(y_train[:,0], train_predictions[:,0])),
                       str(r2_score(y_val[:,0], val_predictions[:,0])), 
                       str(r2_score(y_test[:,0], test_predictions[:,0]))],
                      [str(spearmanr(y_train[:,0], train_predictions[:,0])[0]),
                       str(spearmanr(y_val[:,0], val_predictions[:,0])[0]), 
                       str(spearmanr(y_test[:,0], test_predictions[:,0])[0])]]

      istarr_scores = [[str(r2_score(y_train[:,1], train_predictions[:,1])),
                        str(r2_score(y_val[:,1], val_predictions[:,1])), 
                        str(r2_score(y_test[:,1], test_predictions[:,1]))],
                       [str(spearmanr(y_train[:,1], train_predictions[:,1])[0]),
                        str(spearmanr(y_val[:,1], val_predictions[:,1])[0]), 
                        str(spearmanr(y_test[:,1], test_predictions[:,1])[0])]]

      # write r2 and spearman scores for all of train, test, and val sets and starr & istarr
      write_results_to_file(dir_path+"/results_starr.csv", starr_scores)
      write_results_to_file(dir_path+"/results_istarr.csv", istarr_scores)
      # write_results_to_file(dir_path+"/results_both.csv", both_scores)


    else:
      # calculate all scores from 
      scores = [[str(r2_score(y_train, saved_model.predict(X_train).reshape(1, -1)[0])),
                 str(r2_score(y_val, saved_model.predict(X_val).reshape(1, -1)[0])), 
                 str(r2_score(y_test, saved_model.predict(X_test).reshape(1, -1)[0]))
                 ],
                [str(spearmanr(y_train, saved_model.predict(X_train).reshape(1, -1)[0])[0]),
                 str(spearmanr(y_val, saved_model.predict(X_val).reshape(1, -1)[0])[0]), 
                 str(spearmanr(y_test, saved_model.predict(X_test).reshape(1, -1)[0])[0])
                ]]

      # write r2 and spearman scores for all of train, test, and val sets
      write_results_to_file(dir_path+"/results.csv", scores)


def write_results_to_file(path, scores):
    # creates a file at path address, writes scores from scores nested list to output
    with open(path, "w") as f:
      f.write(",train,val,test\n")
      f.write("r2,"+scores[0][0]+","+scores[0][1]+","+scores[0][2]+"\n")
      f.write("spearman,"+scores[1][0]+","+scores[1][1]+","+scores[1][2]+"\n")


def main():
    args = fetch_args()  # get args

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(args)  # get dataset
    print("got data")

    model = xgboost.XGBRegressor(random_state=1202)  # initalize model

    # create path to folder with results 
    dir_path = ("experiments/xgb"
                +"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                +"_nuc"
                +str(args.include_mononuc_freq)
                +str(args.include_dinuc_freq)
                +str(args.include_trinuc_freq)
                +"_data"
                +str(args.include_starr)
                +str(args.include_istarr))

    os.mkdir(dir_path)

    # train model
    model.fit(X_train, y_train)

    save_results(args, dir_path, X_train, X_test, X_val, y_train, y_test, y_val, model)

    # write all args to text file for reproducibility 
    json.dump(vars(args), open(dir_path+"/settings.txt", "w"))  # https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python


main()