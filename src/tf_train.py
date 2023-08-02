# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import re
import tensorflow as tf
import time
import glob
import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU
from tensorflow.keras.optimizers import RMSprop

# from utils import load_data
from tensorflow.keras import Model, layers

print("TensorFlow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-folder",
    type=str,
    dest="data_folder",
    default="data",
    help="data folder mounting point",
)
parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=128,
    help="mini batch size for training",
)
parser.add_argument(
    "--first-layer-neurons",
    type=int,
    dest="n_hidden_1",
    default=128,
    help="# of neurons in the first layer",
)
parser.add_argument(
    "--second-layer-neurons",
    type=int,
    dest="n_hidden_2",
    default=128,
    help="# of neurons in the second layer",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    dest="learning_rate",
    default=0.01,
    help="learning rate",
)
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="location of the model or checkpoint files from where to resume the training",
)
args = parser.parse_args()

# Start Logging
mlflow.start_run()

# enable autologging
mlflow.tensorflow.autolog()

previous_model_location = args.resume_from
# You can also use environment variable to get the model/checkpoint files location
# previous_model_location = os.path.expandvars(os.getenv("AZUREML_DATAREFERENCE_MODEL_LOCATION", None))

data_folder = args.data_folder
print("Data folder:", data_folder)

from azureml.core import Workspace, Dataset, Datastore
subscription_id = '6bf9b68f-809e-4986-a474-300c6ca2eaa9'
resource_group = 'mcdiismedinarg001'
workspace_name = 'mcdiaml001'

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "workspaceblobstore")
  
dataset01 = Dataset.Tabular.from_delimited_files(path=(datastore, 'mcdi2021-proyecto/data/master_n_notes.csv'))
dataset02 = Dataset.Tabular.from_delimited_files(path=(datastore, 'mcdi2021-proyecto/data/unique_n_notes_sequences.csv'))
master_n_notes = dataset01.to_pandas_dataframe() 
unique_n_notes_sequences = dataset02.to_pandas_dataframe()
master_n_notes = master_n_notes.set_index('Column1')
unique_n_notes_sequences = unique_n_notes_sequences.set_index('Column1')
master_n_notes.index.names = [None]
unique_n_notes_sequences.index.names = [None]

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
## Esta es la definición de un tensor x de una dimensión que almacena el id de cada n-notas
x = np.zeros((unique_n_notes_sequences.shape[0], unique_n_notes_sequences.shape[1] - 2, master_n_notes.shape[0]), dtype=bool)

## Esta es la definición de un tensor x de una dimensión que almacena los ids de las n-notas previa y actual a
##    fin de definir secuencias con 2 grupos de n-notas
#x = np.zeros((unique_n_notes_sequences.shape[0], master_n_notes.shape[0]), dtype=bool)

## Esta es la definición de un tensor y de una dimensión que almacena el id de cada n-nota subsecuente a las x's n-notas previas
y = np.zeros((unique_n_notes_sequences.shape[0], master_n_notes.shape[0]), dtype=bool)

## Aquí se arma el One-Hot encoding para representar los acordes de entrada y de salida
for key, row in unique_n_notes_sequences.iterrows():
    ## Aquí se establece el acorde previo
    #x[key, 0, row[0]] = 1
    ## Aquí se establece el acorde actual
    #x[key, 1, row[1]] = 1
    ## Aquí se establece el acorde siguiente
    #y[key, row[2]] = 1
    
    ## Aquí se establece el acorde previo
    x[key, 0, row[0]] = 1
    ## Aquí se establece el acorde siguiente
    y[key, row[1]] = 1

# Build neural network model.
start_time = time.perf_counter()

modelLSTM = Sequential()
modelLSTM.add(LSTM(1024, input_shape = (x.shape[1:]), dropout = 0.0))
modelLSTM.add(Dense(master_n_notes.shape[0]))
modelLSTM.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
modelLSTM.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = modelLSTM.fit(x, y, validation_split=0.15, batch_size=128, epochs=4, shuffle=True).history

modelGRU = Sequential()
modelGRU.add(GRU(1024, input_shape = (x.shape[1:]), dropout = 0.0))
modelGRU.add(Dense(master_n_notes.shape[0]))
modelGRU.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
modelGRU.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = modelGRU.fit(x, y, validation_split=0.15, batch_size=128, epochs=4, shuffle=True).history

print("LSTM val_accuracy:", modelLSTM.metrics[1].result().numpy())
print("GRU  val_accuracy:", modelGRU.metrics[1].result().numpy())

if modelLSTM.metrics[1].result().numpy() >= modelGRU.metrics[1].result().numpy():
    print("se usará el modelo LSTM")
    model = modelLSTM
else:
    print("se usará el modelo GRU")    
    model = modelGRU

if previous_model_location:
    # Restore variables from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=model_net, optimizer=optimizer)
    checkpoint_file_path = tf.train.latest_checkpoint(previous_model_location)
    checkpoint.restore(checkpoint_file_path)
    checkpoint_filename = os.path.basename(checkpoint_file_path)
    num_found = re.search(r"\d+", checkpoint_filename)
    if num_found:
        start_epoch = int(num_found.group(0))
        print("Resuming from epoch {}".format(str(start_epoch)))

# log accuracies
mlflow.log_metric("LSTM_acc", float(modelLSTM.metrics[1].result().numpy()))
mlflow.log_metric("GRU_acc", float(modelGRU.metrics[1].result().numpy()))

# Save checkpoints in the "./outputs" folder so that they are automatically uploaded into run history.
checkpoint_dir = "./outputs/"
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

mlflow.log_metric("final_acc", float(model.metrics[1].result().numpy()))
os.makedirs("./outputs/model", exist_ok=True)

# files saved in the "./outputs" folder are automatically uploaded into run history
# this is workaround for https://github.com/tensorflow/tensorflow/issues/33913 and will be fixed once we move to >tf2.1
model._set_inputs(x)
tf.saved_model.save(model, "./outputs/model/")

stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))
