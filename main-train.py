import os
import numpy as np
import csv
import pylab as p
import sys
import torch
from train_model import train_model

if __name__ == '__main__':
  # Data from the real user should be labelled 1
  # Data from other users should be labelled 0
  train_data  = np.load('train_mlp_data.npy')
  train_label = np.load('train_mlp_label.npy')
  val_data    = np.load('val_mlp_data.npy')
  val_label   = np.load('val_mlp_label.npy')

  # Hyper-parameters for training
  num_hidden = 100
  learning_rate = 1e-3
  batch_size = 4

  train_model(train_data, train_label, val_data, val_label, \
              num_hidden, learning_rate, batch_size)
