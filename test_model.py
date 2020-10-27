import os
import numpy as np
import csv
import pylab as p
import sys
import time
import torch

def test_model(testing_data, testing_label, num_hidden):
  # Data from the real user should be labelled 0
  # Data from other users should be labelled 1
  assert np.all(np.logical_or(testing_label == 1, testing_label == 0)), \
          'Error: testing_label contains non-zero and non-one labels'

  # labels should be a vector
  assert testing_label.ndim == 1, 'Error: testing_label is not a vector'

  # data should be 2-d matrices and the length of data and label should match
  assert testing_data.ndim == 2, 'Error: testing_data is not a matrix'
  assert testing_data.shape[0] == testing_label.shape[0], \
          'Error: testing_data and testing_label size mismatch'

  testing_label = np.int32(testing_label)
  feature_num = testing_data.shape[1]

  model = torch.load("testing-PC-"+str(num_hidden)+".pt")
  model.eval()

  test_predict = model(torch.from_numpy(testing_data).float())
  score_pred  = test_predict.detach().numpy()
  assert score_pred.shape[0] == testing_data.shape[0]
  assert score_pred.shape[1] == 2
  np.savetxt("testing-PC-"+str(num_hidden)+".csv", score_pred, delimiter = ',')
  score_real  = score_pred[:,0]
  score_other = score_pred[:,1]
  binary_result = score_real < score_other
  ifequal = binary_result == testing_label
  assert ifequal.shape[0] == testing_data.shape[0]
  tn = np.sum(np.logical_and(np.equal(binary_result, testing_label), testing_label == 0))
  tp = np.sum(np.logical_and(np.equal(binary_result, testing_label), testing_label == 1))
  fp = np.sum(np.logical_and(np.not_equal(binary_result, testing_label), testing_label == 0))
  fn = np.sum(np.logical_and(np.not_equal(binary_result, testing_label), testing_label == 1))
  assert tn + fp == np.sum(testing_label == 0)
  assert tp + fn == np.sum(testing_label == 1)

  return tn, tp, fp, fn

if __name__ == '__main__':
  print('Not executable!')
