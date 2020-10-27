import os
import numpy as np
import csv
import pylab as p
import sys
import torch
from net_impl import OneHiddenNet

def train_model(training_data, training_label, val_data, val_label, \
                num_hidden, learning_rate, batch_size):
  # Data from the real user should be labelled 0
  # Data from other users should be labelled 1
  assert np.all(np.logical_or(training_label == 1, training_label == 0)), \
          'Error: training_label contains non-zero and non-one labels'
  assert np.all(np.logical_or(val_label == 1, val_label == 0)), \
          'Error: val_label contains non-zero and non-one labels'

  # labels should be a vector
  assert training_label.ndim == 1, 'Error: training_label is not a vector'
  assert val_label.ndim == 1, 'Error: val_label is not a vector'

  # data should be 2-d matrices and the length of data and label should match
  assert training_data.ndim == 2, 'Error: training_data is not a matrix'
  assert val_data.ndim == 2, 'Error: val_data is not a matrix'
  assert training_data.shape[0] == training_label.shape[0], \
          'Error: training_data and training_label size mismatch'
  assert val_data.shape[0] == val_label.shape[0], \
          'Error: val_data and val_label size mismatch'

  fo = open('training.out', 'w')

  training_label = np.int32(training_label)
  feature_num = training_data.shape[1]

  batch_num = int(training_data.shape[0] / batch_size)
  # Only classify as 'real user' or 'not real user'
  num_classes = 2

  criterion = torch.nn.CrossEntropyLoss()

  print("Number of hidden nodes: "+str(num_hidden))
  print("Input vector size: "+str(feature_num))
  fo.write('Number of hidden nodes: '+str(num_hidden)+'\n')
  fo.write('Input vector size: '+str(feature_num)+'\n')

  model = OneHiddenNet(feature_num, num_hidden, num_classes)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # You can set a different condition to stop training, e.g. using the validation set,
  # other than running a fixed number of rounds
  epochs = 15
  for epoch in range(epochs):
    p = np.random.permutation(training_data.shape[0])
    training_data = training_data[p, :]
    training_label = training_label[p]

    loss_train = []
    accuracy_train = []
    for i in range(batch_num):
      current_data  = training_data[i * batch_size : (i+1) * batch_size, :]
      current_label = training_label[i*batch_size : (i+1)*batch_size].reshape((batch_size))
      y_pred = model(torch.from_numpy(current_data).float())
      loss = criterion(y_pred, torch.from_numpy(current_label).type(torch.LongTensor))
      loss_train.append(np.mean(loss.detach().numpy()))

      # update model parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      score_pred  = y_pred.detach().numpy()
      assert score_pred.shape[0] == batch_size
      assert score_pred.shape[1] == num_classes
      assert num_classes == 2
      # the first score is how likely the sample is from the real user
      score_real = score_pred[:,0]
      # the second score is how likely the sample is from another user (impostor)
      score_other  = score_pred[:,1]
      binary_result = score_real < score_other
      ifequal = binary_result == current_label
      assert ifequal.shape[0] == batch_size
      current_batch_accuracy = np.mean(ifequal)
      accuracy_train.append(current_batch_accuracy)
    print("Epoch "+str(epoch)+"training loss "+str(np.mean(loss_train)))
    print("Epoch "+str(epoch)+"training accuracy "+str(np.mean(accuracy_train)))
    fo.write("Epoch "+str(epoch)+"training loss "+str(np.mean(loss_train))+'\n')
    fo.write("Epoch "+str(epoch)+"training accuracy "+str(np.mean(accuracy_train))+'\n')
  # save the model for testing on PC
  torch.save(model, "testing-PC-"+str(num_hidden)+".pt")
  # save the model for testing on smartphone
  scripted_module = torch.jit.script(model)
  torch.jit.save(scripted_module, "testing-smartphone-"+str(num_hidden)+".pt")

  fo.close()
  print('Training finished!')

if __name__ == '__main__':
  print('Not executable!')

