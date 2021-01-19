import os
import numpy as np
import csv
import pylab as p
import sys
from test_model import test_model

if __name__ == '__main__':
  num_hidden = 100
  print('Number of hidden nodes: '+str(num_hidden))

  testing_data  = np.load('test_mlp_data.npy')
  testing_label = np.load('test_mlp_label.npy')

  tn, tp, fp, fn = test_model(testing_data, testing_label, num_hidden)
  assert tn + fp == np.sum(testing_label == 0)
  assert tp + fn == np.sum(testing_label == 1)

  eps = 1e-12

  fpr = fp / (fp + tn + eps)
  fnr = fn / (fn + tp + eps)
  acc = (tn + tp) / (1. * (tn + tp + fn + fp))
  prec = tp / ( ( tp + fp + eps) * 1. )
  rec =  tp / ( ( tp + fn + eps) * 1. )
  f1 = 2. * prec * rec / ( prec + rec + eps )

  print('----------------Detection Results------------------')
  print('False positives: ', fp)
  print('True negatives: ', tn)
  print('False negatives: ', fn)
  print('True positives: ', tp)
  print('False Positive Rate: ', fpr)
  print('False Negative Rate: ', fnr)
  print('Accuracy: ', acc)
  print('Precision: ', prec)
  print('Recall: ', rec)
  print('F1: ', f1)
  print('---------------------------------------------------')
