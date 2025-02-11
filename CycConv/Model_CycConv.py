#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:28:49 2023

@author: alfonsocabezonvizoso
"""

'''SP using the CycConv approach'''

import time
start = time.time()
#Load encoder
from UP import CyclicPeptide
CP = CyclicPeptide()
# LOAD ml RELATED lIBRARIES
import tensorflow as tf
tf.random.set_seed(42)
from keras.layers import Dense, Input
import keras_tuner as kt
from ConvBlock import ConvBlock
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
                    choices=['AllPep'])
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

X = data['Sequence'].to_numpy()# All except the Permeable column
for idx, peptide in enumerate(X):
    peptide = CP.fragment_peptide(peptide) # Fragment peptide into its monomer blocks
    peptide = ['X', peptide[-1]] + peptide + [peptide[0], 'X']
# =============================================================================
#     The previous step adds the null amino acid X and the last block to the beginning 
#     and the first block and the null amino acid X to the end. This will ensure that
#     after convolution we have the patterns that represent the closure of the CP
# =============================================================================
    X[idx] = ''.join(peptide) # Make peptide whole again
y = data['Permeable'].to_numpy() #Our target
train_portion = 0.8
val_portion = 0.2

# Model builder function
def model_builder(hp):
    '''
    This function creates a Machine Learning model based on sequential
    properties using keras
    Layers:
        1. Two stacked Convolutional Layers and LSTM embedded in the Conv Block
        2. Multi Layer Perceptron
        3. Output layer with sigmoid activation.

    Parameters
    ----------
    hp : keras tuner object
        Object passed internally by the tuner.

    Returns
    -------
    model : keras.src.models.sequential.Sequential
        Sequential Properties model.

    '''
    inputs = Input(shape = (14+4, 208))
    #Add convolutional block
    filters_conv1 = hp.Int('filters_conv1', min_value = 32, max_value = 256, step = 32)
    filters_conv2 = hp.Int('filters_conv2', min_value = 16, max_value = 144, step = 16)
    units_lstm = hp.Int('units_LSTM', min_value = 64, max_value = 256, step = 64)
    name = 'Conv_Block'
    convblock = ConvBlock(filters_conv1, filters_conv2, units_lstm, name = name)(inputs) # Convolutional block
    #Add first layer of the fully connected part
    hp_neurons = hp.Int('units_Dense', min_value=64, max_value = 256, step = 64)
    dense1 = Dense(hp_neurons, activation = 'relu')(convblock)
    #Add a hidden layer
    hp_neurons2 = hp.Int('units_Dense2', min_value = 4, max_value = 132, step = 16)
    dense2 = Dense(hp_neurons2, activation = 'relu')(dense1)
    #Add output layer
    output = Dense(1, activation='sigmoid')(dense2)
    model = tf.keras.Model(inputs=inputs, outputs = output)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-3, 1e-4])
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
    TRAIN = X_train
    TRAIN_target = y_train
    TRAIN, _ = CP.encode(TRAIN, length = 14+4, stop_signal = False)
    TRAIN_length = len(TRAIN)
    #Test
    TEST = X[test]
    TEST_target = y[test]
    TEST, _ = CP.encode(TEST, length = 14+4, stop_signal = False)
    TEST_length = len(TEST)
    #Validation
    VAL = X_val
    VAL_target = y_val
    VAL, _  = CP.encode(VAL, length = 14+4, stop_signal = False)
    VAL_length = len(VAL)
    
    file_keras = f'./trained_models/OptimizedModel_fold{fold_no}.keras'
    file_length = f'{dataset}_CycConv_split_lengths.dat'
    file_acc = f'Accuracy_CycConv_{dataset}.dat'
    file_metric = f'Metrics_CycConv_{dataset}.dat'
    if args.Trained:
        model = tf.keras.models.load_model(file_keras)
    else:
        tuner = kt.Hyperband(model_builder, objective = 'val_accuracy',
                              max_epochs = 80, factor = 3, project_name=f'Tuner_fold{fold_no}', 
                              overwrite=False, seed = 42)
        # Create early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=5)
        
        # Start hyperparameter search
        tuner.search(TRAIN, TRAIN_target, epochs = 2000, validation_data = (VAL, VAL_target),
                      callbacks = [early_stopping], batch_size = 32)
        # Get best tuning model
        model1 = tuner.get_best_models(num_models=1)[0]
        acc1 = model1.evaluate(TEST, TEST_target, verbose = 0)[1]
        
        #Get optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] #returns top performing hyp
        model2 = tuner.hypermodel.build(best_hps) #automatically builds the model with the best parameters
        
        # Train early_stopping
        early_stopping_train = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=35, restore_best_weights=True)
        
        ##TRAIN the model
        history = model2.fit(TRAIN, TRAIN_target, epochs = 2000,
                  validation_data=(VAL, VAL_target), batch_size= 32, verbose = 1,
                  callbacks=[early_stopping_train])
        acc2 = model2.evaluate(TEST, TEST_target, verbose = 0)[1]
        # Decide which model to choose
        if acc1 > acc2:
            model = model1
        else:
            model = model2
        # Save trained model
        file_keras = f'./trained_models/OptimizedModel_fold{fold_no}.keras'
        model.save(file_keras)

    #EVALUATION
    scores_train = model.evaluate(TRAIN, TRAIN_target, verbose = 0)
    scores = model.evaluate(TEST, TEST_target, verbose = 0)
    scores_val = model.evaluate(VAL, VAL_target, verbose = 0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    data_2_plot = [208, fold_no, scores_val[1]*100, scores[1]*100,scores_train[1]*100]
    sep = '    '
    # COMPUTE METRICS
    y_pred = model.predict(TEST)
    new_y = []
    #Round values to be 0 or 1
    for i in y_pred:
        for j in i:
            new_y.append(round(j))
    y_pred = np.array(new_y)
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
            f.write('fold_no'+sep+'TRAIN'+sep+'TEST'+sep+'VAL\n')
        f.write(f'{data_2_plot[1]}{sep}{TRAIN_length}{sep}{TEST_length}{sep}{VAL_length}\n')
        f.close()

    exist_file = os.system(f'ls {file_metric}')
    with open(file_metric, 'a') as f:
        if exist_file != 0:
            f.write('Fold'+sep+'True00'+sep+'False01'+sep+'False10'+sep+'True11'+sep+'Precision'+sep+'f1'+sep+'Recall'+sep+'Roc_auc'+sep+'Matthews'+sep+'Geom_mean\n')
        f.write('{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}\n'.format(fold_no, sep, confusion_TEST[0][0],sep,
                                                                  confusion_TEST[0][1],sep,
                                                                  confusion_TEST[1][0],sep,
                                                                  confusion_TEST[1][1],sep,
                                                                  precision_TEST,sep,f1_TEST,
                                                                  sep,recall_TEST,sep,roc_auc_TEST,
                                                                  sep,matthews_TEST,sep,geom_mean_TEST))
        f.close()

    exist_file = os.system(f'ls {file_acc}')
    with open(file_acc, 'a') as f:
        if exist_file != 0:
            f.write('n_Features'+sep+'fold_no'+sep+'Acc_validation'+sep+'Acc_test'+sep+'Acc_train\n')
        f.write(f'{data_2_plot[0]}{sep}{data_2_plot[1]}{sep}{data_2_plot[2]}{sep}{data_2_plot[3]}{sep}{data_2_plot[4]}\n')
        f.close()
    #Increase fold number
    fold_no += 1
end = time.time()
print(f'Time needed: {end - start}s')
