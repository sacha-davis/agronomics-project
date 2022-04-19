import argparse
import keras
import warnings, logging
import numpy as np
import pandas as pd
import datetime, time, os
import json
import random
import tensorflow as tf

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam  # https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping  # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

from sklearn.metrics import r2_score
from scipy.stats import spearmanr  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

warnings.filterwarnings('ignore')
logging.disable(1000)

tf.random.set_seed(1202)  # https://www.tensorflow.org/api_docs/python/tf/random/set_seed

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
    parser.add_argument('-is', '--include_istarr', type=int, default=0, help='Whether to include the iSTARR target value in the output vector')

    parser.add_argument('-de', '--include_dev', type=int, default=0, help='Whether to include the dev target value in the output vector')
    parser.add_argument('-hk', '--include_hk', type=int, default=0, help='Whether to include the hk target value in the output vector')

    parser.add_argument('-l1', '--layer_1_size', type=int, default=12, help='Num nodes in hidden layer 1')
    parser.add_argument('-a1', '--layer_1_activation', type=str, default="relu", help='Activation for hidden layer 1')
    parser.add_argument('-l2', '--layer_2_size', type=int, default=0, help='Num nodes in hidden layer 2')
    parser.add_argument('-a2', '--layer_2_activation', type=str, default="relu", help='Activation for hidden layer 2')
    parser.add_argument('-l3', '--layer_3_size', type=int, default=0, help='Num nodes in hidden layer 3')
    parser.add_argument('-a3', '--layer_3_activation', type=str, default="relu", help='Activation for hidden layer 3')

    parser.add_argument('-lo', '--output_layer_size', type=int, default=1, help='Num nodes in output layer')
    parser.add_argument('-ao', '--output_layer_activation', type=str, default="linear", help='Activation for output layer')

    args = parser.parse_args()

    return args


def get_model(args, in_dim):  # initializes model architecture
    mdl = Sequential()

    # this is the only layer that is enforced. to test linear regression only, set layer_1_size to 1 and layer_1_activation to "linear"
    mdl.add(Dense(args.layer_1_size, input_dim=in_dim, activation=args.layer_1_activation))

    if args.layer_2_size > 0:       mdl.add(Dense(args.layer_2_size, activation=args.layer_2_activation))
    if args.layer_3_size > 0:       mdl.add(Dense(args.layer_3_size, activation=args.layer_3_activation))
    if args.output_layer_size > 0:  mdl.add(Dense(args.output_layer_size, activation=args.output_layer_activation))

    return mdl


def return_y(args, df):  # based on what to include, returns y array
    
    # TEMPORARY -- TODO: FIX THIS SO AUX IS POSSIBLE
    if args.include_starr:
      y = np.array(df["target_starr"].tolist())
    elif args.include_istarr:
      y = np.array(df["target_istarr"].tolist())
    elif args.include_dev:
      y = np.array(df["dev_target"].tolist())
    else: #args.include_hk
      y = np.array(df["hk_target"].tolist())

    # if args.include_starr:
    #   if args.include_istarr:
    #     y = np.array(pd.concat([df["target_starr"], df["target_istarr"]], axis=1))
    #   else:
    #     y = np.array(df["target_starr"].tolist())
    # else:
    #   y = np.array(df["target_istarr"].tolist())

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
      # TODO: will need to be fixed in future
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

      # both_scores = [[str(r2_score(y_train, train_predictions)),
      #                 str(r2_score(y_val, val_predictions)), 
      #                 str(r2_score(y_test, test_predictions))],
      #                [str(spearmanr(y_train, train_predictions)[0]),
      #                 str(spearmanr(y_val, val_predictions)[0]), 
      #                 str(spearmanr(y_test, test_predictions)[0])]]

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

    model = get_model(args, X_train.shape[1])  # initalize model

    # create path to folder with results 
    dir_path = ("experiments/dros"
                +"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                +"_nuc"
                +str(args.include_mononuc_freq)
                +str(args.include_dinuc_freq)
                +str(args.include_trinuc_freq)
                +"_lay"+str(args.layer_1_size)
                +"-"+str(args.layer_2_size)
                +"-"+str(args.layer_3_size)
                +"-"+str(args.output_layer_size)
                +"_lr"+str(args.learning_rate)
                +"_bs"+str(args.batch_size))

    # compile model
    model.compile(optimizer=Adam(lr=args.learning_rate),  # CHANGE IF WE WANT TO CHANGE OPTIM
                  loss='mean_squared_error')

    # init callbacks
    logdir = os.path.join(dir_path, "logs")
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)  # https://stackoverflow.com/questions/59894720/keras-and-tensorboard-attributeerror-sequential-object-has-no-attribute-g
    es_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
    mc_callback = ModelCheckpoint(dir_path+'/best_model.h5', monitor='val_loss', save_best_only=True)

    # train model
    history = model.fit(X_train, y_train,
                        epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard_callback, es_callback, mc_callback])

    # save training history
    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(dir_path+'/training_history.csv')

    # load best model according to val_loss
    saved_model = load_model(dir_path+'/best_model.h5')

    save_results(args, dir_path, X_train, X_test, X_val, y_train, y_test, y_val, saved_model)

    # write all args to text file for reproducibility 
    json.dump(vars(args), open(dir_path+"/settings.txt", "w"))  # https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python



main()

