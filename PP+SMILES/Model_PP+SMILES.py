#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:40:21 2024

@author: alfonsocabezonvizoso
"""

'''SP+SMILES model.'''

import time
start = time.time()
#Load encoder
from encoder_PCA import Encoder
encoder = Encoder(properties='All')
from smi_encoder import Smi_encoder
smi_encode = Smi_encoder()
# LOAD ml RELATED lIBRARIES
import tensorflow as tf
from keras.layers import Dense, Input, Concatenate
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

###############################################################################

dataset = 'AllPep' # get dataset name
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
    This function creates a Machine Learning model based on the combination of 
    peptide properties and SMILES using keras
    Layers:
        1. Two input layers for the two different inputs
        2. Concatenation to join PP and SMILE
        3. Multi Layer Perceptron
        4. Output layer with sigmoid activation.

    Parameters
    ----------
    hp : keras tuner object
        Object passed internally by the tuner.

    Returns
    -------
    model : keras.src.models.sequential.Sequential
        PP+SMILE model.

    '''
    #define sets of inputs
    input_Seq_prop = Input(shape=(208,)) # Shape 16820, 4)
    input_SMILE = Input(shape=(163,))
    # CONCATENATE THE INPUT FROM THE SMILE WITH THE Input of SP
    concatenation = Concatenate(axis=-1)([input_Seq_prop, input_SMILE])
    #Add first layer of the fully connected part
    hp_neurons = hp.Int('units_Dense', min_value=16, max_value = 208, step = 16)
    Dense1 = Dense(hp_neurons, activation = 'relu')(concatenation)
    #Add a hidden layer
    hp_neurons2 = hp.Int('units_Dense2', min_value = 4, max_value = 132, step = 16)
    Dense2 = Dense(hp_neurons2, activation = 'relu')(Dense1)
    #Add output layer
    output_layer = Dense(1, activation='sigmoid')(Dense2)
    model = tf.keras.Model(inputs=[input_Seq_prop, input_SMILE], outputs = output_layer)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-3, 1e-4])
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
    TRAIN_SMILE = smi_encode.encode_smi(TRAIN)
    _, _ , TRAIN_seq_prop= encoder.encode(sequences = TRAIN, length = 15,
                                          stop_signal = False,
                                          sequence_properties = True)
    #Test
    TEST = X[test]
    TEST_length = len(TEST)
    TEST_target = y[test]
    TEST_SMILE = smi_encode.encode_smi(TEST)
    _, _, TEST_seq_prop = encoder.encode(sequences = TEST, length = 15,
                                            stop_signal = False, 
                                            sequence_properties = True)
    #Validation
    VAL = X_val
    VAL_length = len(VAL)
    VAL_target = y_val
    VAL_SMILE = smi_encode.encode_smi(VAL)
    _, _, VAL_seq_prop = encoder.encode(sequences = VAL, length = 15,
                                            stop_signal = False, sequence_properties = True)
    
    
    # Instantiate the tuner for hyperband search
    tuner = kt.Hyperband(model_builder, objective = 'val_accuracy',
                          max_epochs = 80, factor = 3, project_name=f'Tuner_fold{fold_no}', 
                          overwrite=False, seed = 42)
    # Create early stopping
    early_stopping_tun = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5)

    early_stopping_train = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=35, restore_best_weights=True)
    
    # Start hyperparameter search
    tuner.search([TRAIN_seq_prop, TRAIN_SMILE], TRAIN_target, epochs = 2000,
                 validation_data = ([VAL_seq_prop, VAL_SMILE], VAL_target),
                 callbacks = [early_stopping_tun], batch_size = 32)
    #Get optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] #returns top performing hyp
    #build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps) #automatically builds the model with the best parameters
    ##TRAIN the model
    history = model.fit([TRAIN_seq_prop, TRAIN_SMILE], TRAIN_target, epochs = 2000,
              validation_data=([VAL_seq_prop, VAL_SMILE], VAL_target), batch_size= 32, verbose = 1,
              callbacks=[early_stopping_train])
    # Save trained model
    model.save(f'OptimizedModel_fold{fold_no}.keras')
    #batch size denotes the number of sequence analyze prior to update of internal parameters
    model = tuner.get_best_models(num_models = 1)[0]
    #EVALUATION
    scores_train = model.evaluate([TRAIN_seq_prop, TRAIN_SMILE], TRAIN_target, verbose = 0)
    scores = model.evaluate([TEST_seq_prop, TEST_SMILE], TEST_target, verbose = 0)
    scores_val = model.evaluate([VAL_seq_prop, VAL_SMILE], VAL_target, verbose = 0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    data_2_plot = [208, fold_no, scores_val[1]*100, scores[1]*100,scores_train[1]*100]
    sep = '    '
    # METRICS
    y_pred = model.predict([TEST_seq_prop, TEST_SMILE])
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
    with open(f'{dataset}_split_lengths.dat', 'a') as f:
        if fold_no == 1:
            f.write('fold_no'+sep+'TRAIN'+sep+'TEST'+sep+'VAL\n')
        f.write(f'{data_2_plot[1]}{sep}{TRAIN_length}{sep}{TEST_length}{sep}{VAL_length}\n')
        f.close()

    with open(f'Metrics_{dataset}.dat', 'a') as f:
        if fold_no == 1:
            f.write('Fold'+sep+'True00'+sep+'False01'+sep+'False10'+sep+'True11'+sep+'Precision'+sep+'f1'+sep+'Recall'+sep+'Roc_auc'+sep+'Matthews'+sep+'Geom_mean\n')
        f.write('{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}\n'.format(fold_no, sep, confusion_TEST[0][0],sep,
                                                                  confusion_TEST[0][1],sep,
                                                                  confusion_TEST[1][0],sep,
                                                                  confusion_TEST[1][1],sep,
                                                                  precision_TEST,sep,f1_TEST,
                                                                  sep,recall_TEST,sep,roc_auc_TEST,
                                                                  sep,matthews_TEST,sep,geom_mean_TEST))
        f.close()

    with open(f'Accuracy_{dataset}.dat', 'a') as f:
        if fold_no == 1:
            f.write('n_Features'+sep+'fold_no'+sep+'Acc_validation'+sep+'Acc_test'+sep+'Acc_train\n')
        f.write(f'{data_2_plot[0]}{sep}{data_2_plot[1]}{sep}{data_2_plot[2]}{sep}{data_2_plot[3]}{sep}{data_2_plot[4]}\n')
        f.close()
    #Increase fold number
    fold_no += 1
end = time.time()
print(f'Time needed: {end - start}s')
