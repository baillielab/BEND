"""
bend.models
===========

This module contains the implementations of the supervised models used in the paper.
"""

from .awd_lstm import AWDLSTMConfig, AWDLSTMForLM, AWDLSTMModelForInference
from .dilated_cnn import ConvNetConfig, ConvNetForMaskedLM, ConvNetModel
