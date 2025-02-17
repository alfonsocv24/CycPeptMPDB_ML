#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:37:20 2024

@author: alfonsocabezonvizoso
"""

import keras
from keras import layers


class ConvBlock(layers.Layer):
    def __init__(self, filters_conv1, filters_conv2, units_lstm, neurons_Dense, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters_conv1 = filters_conv1
        self.filters_conv2 = filters_conv2
        self.units_lstm = units_lstm
        # self.n_blocks = n_blocks
        self.drop_rate = drop_rate
        self.neurons_Dense = neurons_Dense
# =============================================================================
#         Create layers
# =============================================================================
        self.Conv1 = layers.Conv1D(self.filters_conv1, kernel_size = 3, 
                                   kernel_initializer = "he_normal")
        self.Conv2 = layers.Conv1D(self.filters_conv2, kernel_size = 3,
                                   kernel_initializer = "he_normal")
        self.LSTM = layers.LSTM(self.units_lstm, activation = 'relu', 
                                return_sequences = False)
        self.Dropout = layers.Dropout(drop_rate)
        self.Dense = layers.Dense(neurons_Dense, activation = 'relu')
        
    def call(self, inputs):
        '''
        Function that creates the convolutional block and feeds it with input

        Parameters
        ----------
        inputs : keras.src.engine.keras_tensor.KerasTensor
            Input tensor to feed Conv block.

        Returns
        -------
        lstm : list of keras.src.engine.keras_tensor.KerasTensor
            List of outputs of the lstms layers.

        '''
        # output = []
        # for i in range(self.n_blocks):
        #     conv1 = self.Conv1(inputs[:, i]) # First Convolution for input slice
        #     dout1 = self.Dropout(conv1) # Dropout first conv
        #     conv2 = self.Conv2(dout1) # Second convolution
        #     dout2 = self.Dropout(conv2) # Dropout second convolution
        #     lstm = self.LSTM(dout2) # LSTM layer for final output
        #     output.append(lstm)
        conv1 = self.Conv1(inputs) # First Convolution for input slice
        # dout1 = self.Dropout(conv1) # Dropout first conv
        conv2 = self.Conv2(conv1) # Second convolution
        # dout2 = self.Dropout(conv2) # Dropout second convolution
        lstm = self.LSTM(conv2) # LSTM layer for final output
        dout = self.Dropout(lstm) # Dropout lstm
        dense = self.Dense(dout)
        return dense
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters_conv1": self.filters_conv1,
            "filters_conv2": self.filters_conv2,
            "units_lstm": self.units_lstm,
            "neurons_Dense" : self.neurons_Dense,
            "drop_rate" : self.drop_rate,
        })
        return config
