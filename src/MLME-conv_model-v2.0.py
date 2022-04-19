import argparse
import keras
import warnings, logging
import numpy as np
import pandas as pd
import datetime, os
import json
import random
import tensorflow as tf

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, BatchNormalization
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam  # https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping  # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

from sklearn.metrics import r2_score
from scipy.stats import spearmanr  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

warnings.filterwarnings('ignore')
logging.disable(1000)

tf.random.set_seed(1202)  # https://www.tensorflow.org/api_docs/python/tf/random/set_seed

# config = tf.compat.v1.ConfigProto()  # https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

mapping = {"A": [1, 0, 0, 0], "T": [0, 0, 0, 1], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0]}  # cross referenced with kipoi data loader


def fetch_args():
    parser = argparse.ArgumentParser(description="Conv Model")

    parser.add_argument('-bi', '--batch_iterator', type=int, default=0, help='Whether to try batch generator or not')

    parser.add_argument('-da', '--data_path', type=str, default="data/processed/hidra_chloroplast_70.csv", help='Whole dataset path')

    parser.add_argument('-tr', '--train_path', type=str, default="data/processed/arabidopsis_train.csv", help='Train dataset path')
    parser.add_argument('-va', '--val_path', type=str, default="data/processed/arabidopsis_val.csv", help='Train dataset path')
    parser.add_argument('-te', '--test_path', type=str, default="data/processed/arabidopsis_test.csv", help='Train dataset path')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('-ep', '--num_epochs', type=int, default=100, help='Total Number of Epochs')
    parser.add_argument('-pa', '--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('-ms', '--max_batch_steps', type=int, default=-1, help='Maximum number of steps in a batch, by default goes through entire batch')
    parser.add_argument('-op', '--optimizer', type=str, default='adam', help='Keras optimizer -- pick from [adam]')
    parser.add_argument('-vt', '--verbose_training', type=int, default=1, help='Verbose Training')

    parser.add_argument('-sl', '--input_sequence_length', type=int, default=145, help='Length of input sequence to the model')
    parser.add_argument('-no', '--number_of_outputs', type=int, default=1, help='Number of outputs of the model')

    parser.add_argument('-c1', '--conv_one_set', type=int, default=0, help='Treatment for first conv layer. 0 = scratch, 1 = starting point, 2 = freeze')
    parser.add_argument('-c2', '--conv_two_set', type=int, default=0, help='Treatment for second conv layer. 0 = scratch, 1 = starting point, 2 = freeze')
    parser.add_argument('-c3', '--conv_three_set', type=int, default=0, help='Treatment for third conv layer. 0 = scratch, 1 = starting point, 2 = freeze')

    parser.add_argument('-mp', '--model_path', type=str, default='models/model.json', help='Path to MPRA-DragoNN model')
    parser.add_argument('-wp', '--weights_path', type=str, default='models/pretrained.hdf5', help='Path to MPRA-DragoNN weights')

    parser.add_argument('-fn', '--fold_num', type=str, default='15', help='Dataset to use')
    parser.add_argument('-lm', '--linear_mapping', type=int, default=0, help='1 to build on top of model output, 0 to replace')
    parser.add_argument('-lc', '--last_conv_layer', type=int, default=1, help='1 to keep last conv layer, 0 to delete it')
    parser.add_argument('-sh', '--shuffle', type=int, default=0, help='1 to shuffle input sequences, 0 to not')

    args = parser.parse_args()

    return args


def train_test_val(df_path, shuffle):  # splits dataframe into all the sets
    df = pd.read_csv(df_path)

    print(df.shape)

    # # uncomment to confirm shufling takes place
    # print(shuffle)
    # print(df["sequence"].head())

    if shuffle == 1:  # shuffles NTs within each sequence
      df.loc[:,"sequence"] = [''.join(random.sample(s, len(s))) for s in df["sequence"]]

    # print(df["sequence"].head())

    train_df = df[df.set == "train"]
    # X_train = np.array(train_df["sequence"].apply(get_ohe))
    X_train = np.array([get_ohe(sqnc) for sqnc in train_df["sequence"]])
    y_train = np.array(train_df["target"].tolist())
    print("done train")

    val_df = df[df.set == "val"]
    X_val = np.array([get_ohe(sqnc) for sqnc in val_df["sequence"]])
    y_val = np.array(val_df["target"].tolist())

    test_df = df[df.set == "test"]
    X_test = np.array([get_ohe(sqnc) for sqnc in test_df["sequence"]])
    y_test = np.array(test_df["target"].tolist())

    return X_train, y_train, X_val, y_val, X_test, y_test


def batch_generator(Train_df,batch_size,steps):  # https://medium.com/analytics-vidhya/train-keras-model-with-large-dataset-batch-training-6b3099fdf366
    idx=1
    while True: 
        yield load_data(Train_df,idx-1,batch_size) ## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1


def load_data(Train_df,idx,batch_size):
    df = pd.read_csv(Train_df, skiprows=idx*batch_size, nrows=batch_size)

    X = np.array([get_ohe(sqnc) for sqnc in df.iloc[:,3]])  # sequence
    y = np.array(df.iloc[:,8])  # target

    return X, y


def load_pretrained_model(model_path, weights_path):  # fetch MPRA-DragoNN
    with open(model_path, 'r') as json_file:
        json_savedModel = json_file.read()  # read in json model architecture

    pretrained_mdl = model_from_json(json_savedModel)  # convert json to usable model format
    pretrained_mdl.load_weights(weights_path)  # load pre-trained weights to 

    return pretrained_mdl


def get_ohe(sequence):  # gets sequence in format model can use (145, 4)
    return np.array([mapping[nt] for nt in sequence])


def get_model(args):  # initializes model architecture
    mdl = Sequential()

    conv1_train = args.conv_one_set != 2  # True if conv layer should be trained
    mdl.add(Conv1D(120, 5, activation='relu', input_shape=(145, 4), name="1DConv_1", trainable=conv1_train))
    mdl.add(BatchNormalization(name="batchNorm1", trainable=conv1_train))
    mdl.add(Dropout(0.1, name="drop1"))

    conv2_train = args.conv_two_set != 2  # True if conv layer should be trained
    mdl.add(Conv1D(120, 5, activation='relu', name="1DConv_2", trainable=conv2_train))
    mdl.add(BatchNormalization(name="batchNorm2", trainable=conv2_train))
    mdl.add(Dropout(0.1, name="drop2"))

    if args.last_conv_layer == 1:  # if we are not removing last conv layer for simplicity
      conv3_train = args.conv_three_set != 2  # True if conv layer should be trained
      mdl.add(Conv1D(120, 5, activation='relu', name="1DConv_3", trainable=conv3_train))
      mdl.add(BatchNormalization(name="batchNorm3", trainable=conv3_train))
      mdl.add(Dropout(0.1, name="drop3"))

    mdl.add(Flatten(name="flat"))

    if args.linear_mapping == 1: 
        mdl.add(Dense(12, activation='linear', name="dense1", trainable=False))

    # output layer
    mdl.add(Dense(1, activation='linear', name="dense2"))

    return mdl


def set_weights(args, pretrained_mdl, mdl):  # sets appropriate model weights from pretrained
    layers_to_set = []  # contains indices of layers to set
    if args.conv_one_set != 0: layers_to_set += [0, 1, 2]
    if args.conv_two_set != 0: layers_to_set += [3, 4, 5]
    if args.conv_three_set != 0: layers_to_set += [6, 7, 8]

    for i in layers_to_set:
        pretrained_layer_weights = pretrained_mdl.layers[i].get_weights()  # get pre-trained layer weights
        mdl.layers[i].set_weights(pretrained_layer_weights)  # set layer weights

    return mdl


def main():
    args = fetch_args()  # get args

    if args.batch_iterator == 1:
      # define generators for batch processing
      train_size = pd.read_csv(args.train_path).shape[0]
      val_size = pd.read_csv(args.val_path).shape[0]
      print(train_size, val_size)

      steps_per_epoch = np.ceil(train_size/args.batch_size)
      validation_steps = np.ceil(val_size/args.batch_size)

      my_training_batch_generator = batch_generator(args.train_path, args.batch_size, steps_per_epoch)
      my_validation_batch_generator = batch_generator(args.val_path, args.batch_size, validation_steps)
    else:
      X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(args.data_path, args.shuffle)  # get dataset

    # models
    pretrained_model = load_pretrained_model(args.model_path, args.weights_path)  # load in MPRA-DragoNN model arch and weights
    model = set_weights(args, pretrained_model, get_model(args))  # instantiate and init model

    # create path to folder with results 
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    arch_settings = str(args.conv_one_set)+str(args.conv_two_set)+str(args.conv_three_set)+str(args.linear_mapping)
    dir_path = "experiments/exp_"+date+"_"+arch_settings+"_lr"+str(args.learning_rate)+"_bs"+str(args.batch_size)+"_ep"+str(args.num_epochs)

    # compile model
    model.compile(optimizer=Adam(lr=args.learning_rate),  # CHANGE IF WE WANT TO CHANGE OPTIM
                  loss='mean_squared_error')

    # init callbacks
    logdir = os.path.join(dir_path, "logs")
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)  # https://stackoverflow.com/questions/59894720/keras-and-tensorboard-attributeerror-sequential-object-has-no-attribute-g
    es_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience, min_delta=0.001)
    mc_callback = ModelCheckpoint(dir_path+'/best_model.h5', monitor='val_loss', save_best_only=True)

    # train model
    if args.batch_iterator == 1:  # if we want to use the batch iterator
      history = model.fit_generator(my_training_batch_generator,
                                    epochs=args.num_epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    verbose=1, 
                                    validation_data=my_validation_batch_generator,
                                    validation_steps=validation_steps, 
                                    callbacks=[tensorboard_callback, es_callback, mc_callback])
    else:  # if we want to just pass in the data variables
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

    if args.batch_iterator == 1:
      X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(args.data_path, args.shuffle)  # get datasets for calculating scores

    # write r2 and spearman scores for all of train, test, and val sets
    with open(dir_path+"/results.csv", "w") as f:
      f.write(",R2,spearman\n")
      f.write("train,"+str(r2_score(y_train, saved_model.predict(X_train).reshape(1, -1)[0]))+","+str(spearmanr(y_train, saved_model.predict(X_train).reshape(1, -1)[0])[0])+"\n")
      f.write("val,"+str(r2_score(y_val, saved_model.predict(X_val).reshape(1, -1)[0]))+","+str(spearmanr(y_val, saved_model.predict(X_val).reshape(1, -1)[0])[0])+"\n")
      f.write("test,"+str(r2_score(y_test, saved_model.predict(X_test).reshape(1, -1)[0]))+","+str(spearmanr(y_test, saved_model.predict(X_test).reshape(1, -1)[0])[0]))

    # write all args to text file for reproducibility 
    json.dump(vars(args), open(dir_path+"/settings.txt", "w"))  # https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python


main()
