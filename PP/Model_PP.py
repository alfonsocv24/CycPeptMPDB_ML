#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:28:49 2023

@author: alfonsocabezonvizoso
"""

'''This script is an adaptation of the ML code developed in Rijeka. In this case,
we are combining Sequential Properties + Sequence Properties + SMILE embedding.'''

import time
start = time.time()
#Load encoder
from encoder_PCA import Encoder
encoder = Encoder(properties='All')
# LOAD ml RELATED lIBRARIES
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
import keras_tuner as kt
# Load sklearn libraries
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # Initialize scaler
# Load data processing libraries
import pandas as pd
import numpy as np
# General Purpose Libraries
import sys
import argparse
import os

def geometric_mean_score(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr * tnr)**0.5

'''Define user inputs'''
###############################################################################
parser = argparse.ArgumentParser(description='''ML code''')
parser.add_argument('-ds', '--DATASET', help='Dataset name, AllPep or L67',
                    action='store', type=str, default = 'AllPep',
                    choices=['AllPep', 'L67'])
parser.add_argument('-t', '--Trained', help = 'Use trained model',
                    action = 'store_true')

args = parser.parse_args()
###############################################################################
dataset = args.DATASET # get dataset name
file = f'CycPeptMPDB_{dataset}.csv'
#Load data

data = pd.read_csv(file, header = 0)
#Assign a 1 to permeable if permeability < -6, otherwise assign 0
data['Permeable'] = np.where(data['Permeability'] < -6, 1, 0 )

# Eliminate repeated sequences
data.drop_duplicates(subset='Sequence',keep = False,  inplace = True)

X = data['Sequence'].to_numpy()# Get sequences
y = data['Permeable'].to_numpy() #Our target
train_portion = 0.8
val_portion = 0.2

def model_builder(hp):
    '''
    This function creates a Machine Learning model based on peptide
    properties using keras
    Layers:
        1. Multi Layer Perceptron
        2. Output layer with sigmoid activation.

    Parameters
    ----------
    hp : keras tuner object
        Object passed internally by the tuner.

    Returns
    -------
    model : keras.src.models.sequential.Sequential
        Peptide Properties model.

    '''
    #define sets of inputs
    input_Seq_prop = Input(shape=(208,)) # Sequence Properties input
    #Add first layer of the fully connected part
    hp_neurons = hp.Int('units_Dense', min_value=16, max_value = 208, step = 16)
    Dense1 = Dense(hp_neurons, input_shape = (208,), activation = 'relu')(input_Seq_prop) # INPUT FOR THIS LAYER IS THE CONCATENATION
    #Add a second dropout layer
    DropLayer = Dropout(0.2)(Dense1)
    #Add a hidden layer
    hp_neurons2 = hp.Int('units_Dense2', min_value = 4, max_value = 132, step = 16)
    Dense2 = Dense(hp_neurons2, activation = 'relu')(DropLayer)
    #Add output layer
    output_layer = Dense(1, activation='sigmoid')(Dense2)
    model = tf.keras.Model(inputs = input_Seq_prop, outputs = output_layer)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-3, 1e-4])
    #Here the tuner will test the different values we propose
    #define optimizer applying tuner for learning rate
    opt = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
    #Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return model

num_folds = 10
#Define per-fold score containers
acc_per_fold = []
loss_per_fold = []


# Define stratified K-fold cross validator
kfold = RepeatedStratifiedKFold(n_repeats=3, n_splits = num_folds, random_state=42)


fold_no = 1 
for train, test in kfold.split(X, y):
    #Create validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X[train],
                                                    y[train],
                                                    test_size =val_portion,
                                                    random_state=42,
                                                    stratify = y[train])
    '''Perform Data augmentation on test, train and validation sets'''
    
    #Train
    TRAIN = np.copy(X_train) # Training sequences
    TRAIN_length = len(TRAIN)
    TRAIN_target = np.copy(y_train) # Target values of the TRAINING sequences
    _, _ , TRAIN_seq_prop= encoder.encode(sequences = TRAIN, length = 15,
                                              stop_signal = False,
                                              sequence_properties = True)
    #Test
    TEST = X[test]
    TEST_length = len(TEST)
    TEST_target = y[test]
    _, _, TEST_seq_prop = encoder.encode(sequences = TEST, length = 15,
                                            stop_signal = False,
                                            sequence_properties = True)
    #Validation
    VAL = X_val
    VAL_length = len(VAL)
    VAL_target = y_val
    _, _, VAL_seq_prop = encoder.encode(sequences = VAL, length = 15,
                                            stop_signal = False, 
                                            sequence_properties = True)
    
    file_keras = f'./trained_models/OptimizedModel_fold{fold_no}.keras'
    file_length = f'{dataset}_PP_split_lengths.dat'
    file_acc = f'Accuracy_PP_{dataset}.dat'
    file_metric = f'Metrics_PP_{dataset}.dat'
    if args.Trained:
        model = tf.keras.models.load_model(file_keras)
    
    else:
        # Instantiate the tuner for hyperband search
        tuner = kt.Hyperband(model_builder, objective = 'val_accuracy',
                              max_epochs = 80, factor = 3, project_name=f'Tuner_fold{fold_no}', 
                              overwrite=False, seed = 42)
        # Create early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=5)
    
        early_stopping_train = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=35, restore_best_weights=True)
        
        # Start hyperparameter search
        tuner.search(TRAIN_seq_prop, TRAIN_target, epochs = 2000, validation_data = (VAL_seq_prop, VAL_target),
                      callbacks = [early_stopping], batch_size = 32)
        # Get best tuning model
        model1 = tuner.get_best_models(num_models=1)[0]
        acc1 = model1.evaluate(TEST_seq_prop, TEST_target, verbose = 0)[1]
        
        #Get optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] #returns top performing hyp
        model2 = tuner.hypermodel.build(best_hps) #automatically builds the model with the best parameters
        
        # Train early_stopping
        early_stopping_train = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=35, restore_best_weights=True)
        
        ##TRAIN the model
        history = model2.fit(TRAIN_seq_prop, TRAIN_target, epochs = 2000,
                  validation_data=(VAL_seq_prop, VAL_target), batch_size= 32, verbose = 1,
                  callbacks=[early_stopping_train])
        acc2 = model2.evaluate(TEST_seq_prop, TEST_target, verbose = 0)[1]
        # Decide which model to choose
        if acc1 > acc2:
            model = model1
        else:
            model = model2
        # Save trained model
        file_keras = f'./trained_models/OptimizedModel_fold{fold_no}.keras'
        model.save(file_keras)

    #EVALUATION
    scores_train = model.evaluate(TRAIN_seq_prop, TRAIN_target, verbose = 0)
    scores = model.evaluate(TEST_seq_prop, TEST_target, verbose = 0)
    scores_val = model.evaluate(VAL_seq_prop, VAL_target, verbose = 0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    data_2_plot = [208, fold_no, scores_val[1]*100, scores[1]*100,scores_train[1]*100]
    sep = '    '
    # COMPUTE METRICS
    y_pred = model.predict(TEST_seq_prop)
    new_y = []
    #Round values to be 0 or 1
    for i in y_pred:
        for j in i:
            new_y.append(round(j))
    y_pred = np.array(new_y)
    y_pred = 1 - y_pred # Change sign to match CycPeptMPDB
    TEST_target = 1 - TEST_target # Change sign to match CycPeptMPDB
    confusion_TEST = confusion_matrix(TEST_target, y_pred)
    precision_TEST = precision_score(TEST_target, y_pred)
    f1_TEST = f1_score(TEST_target, y_pred)
    recall_TEST = recall_score(TEST_target, y_pred)
    roc_auc_TEST = roc_auc_score(TEST_target, y_pred)
    matthews_TEST = matthews_corrcoef(TEST_target, y_pred)
    geom_mean_TEST = geometric_mean_score(TEST_target, y_pred)
    
    exist_file = os.system(f'ls {file_length}')
    with open(file_length, 'a') as f:
        if exist_file != 0:
            f.write('fold'+sep+'TRAIN'+sep+'TEST'+sep+'VAL\n')
        f.write(f'{data_2_plot[1]}{sep}{TRAIN_length}{sep}{TEST_length}{sep}{VAL_length}\n')
        f.close()

    exist_file = os.system(f'ls {file_metric}')
    with open(file_metric, 'a') as f:
        if exist_file != 0:
            f.write('Fold'+sep+'True00'+sep+'False01'+sep+'False10'+sep+'True11'+sep+'Accuracy'+sep+'Precision'+sep+'f1'+sep+'Recall'+sep+'Roc_auc'+sep+'Matthews'+sep+'Geom_mean\n')
        f.write('{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}\n'.format(fold_no, sep, confusion_TEST[0][0],sep,
                                                                  confusion_TEST[0][1],sep,
                                                                  confusion_TEST[1][0],sep,
                                                                  confusion_TEST[1][1],sep, data_2_plot[3], sep,
                                                                  precision_TEST,sep,f1_TEST,
                                                                  sep,recall_TEST,sep,roc_auc_TEST,
                                                                  sep,matthews_TEST,sep,geom_mean_TEST))
        f.close()

    exist_file = os.system(f'ls {file_acc}')
    with open(file_acc, 'a') as f:
        if exist_file != 0:
            f.write('n_Features'+sep+'fold'+sep+'Acc_validation'+sep+'Acc_test'+sep+'Acc_train\n')
        f.write(f'{data_2_plot[0]}{sep}{data_2_plot[1]}{sep}{data_2_plot[2]}{sep}{data_2_plot[3]}{sep}{data_2_plot[4]}\n')
        f.close()
    #Increase fold number
    fold_no += 1
    
metrics_file = f'{file_metric}'
cols = ['Accuracy', 'Precision', 'f1', 'Recall', 'Roc_auc', 'Matthews', 'Geom_mean']
metrics_all = '/home/ciqus/Scripts/FinalModel/4GitHub/Metrics_ALL.csv'
df = pd.read_csv(metrics_file, sep = '\s+')
df = df[cols]
means = df.mean().to_numpy()
stds = df.std().to_numpy()
plus_minus = "\u00B1"
# Store mean values

model = ['PP']

exist_file = os.system(f'ls {metrics_all}')
with open(metrics_all, 'a') as f:
    if exist_file!= 0:
        header = ['Model'] + cols
        header = "".join(f'{col:<25}' for col in header)
        f.write(header + '\n')
    metrics = [f'{mean:.3f} {plus_minus} {stds[idx]:.3f}' for idx, mean in enumerate(means)]
    line = model + metrics
    line = "".join(f'{col:<25}' for col in line)
    f.write(line + '\n')
    f.close()

end = time.time()
print(f'Time needed: {end - start}s')
