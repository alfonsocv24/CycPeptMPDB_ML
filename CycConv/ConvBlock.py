#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:37:20 2024

@author: alfonsocabezonvizoso
"""

import keras
from keras import layers


class ConvBlock(layers.Layer):
    def __init__(self, filters_conv1, filters_conv2, units_lstm,  **kwargs):
        super().__init__(**kwargs)
        self.filters_conv1 = filters_conv1
        self.filters_conv2 = filters_conv2
        self.units_lstm = units_lstm
# =============================================================================
#         Create layers
# =============================================================================
        self.Conv1 = layers.Conv1D(self.filters_conv1, kernel_size = 3, 
                                   kernel_initializer = "he_normal")
        self.Conv2 = layers.Conv1D(self.filters_conv2, kernel_size = 3,
                                   kernel_initializer = "he_normal")
        self.LSTM = layers.LSTM(self.units_lstm, activation = 'relu', 
                                return_sequences = False)
        
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
        conv1 = self.Conv1(inputs) # First Convolution for input slice
        conv2 = self.Conv2(conv1) # Second convolution
        # dout2 = self.Dropout(conv2) # Dropout second convolution
        lstm = self.LSTM(conv2) # LSTM layer for final output
        return lstm
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters_conv1": self.filters_conv1,
            "filters_conv2": self.filters_conv2,
            "units_lstm": self.units_lstm,
        })
        return config
