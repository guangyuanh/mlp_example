import sys
import numpy as np

def seq_win_vectorize(seq, window_size):
  res = []
  for i in range(len(seq)-window_size+1):
    res.append(seq[i: i+window_size,:].reshape((-1)))
  return np.array(res)

if __name__ == '__main__':
  assert len(sys.argv) >= 2, 'You need to specify an input series csv file!'
  f_name = sys.argv[1]
  file_name_split = f_name.split('.')
  assert file_name_split[-1] == 'csv', 'The input series should be a csv file!'
  print('Input series csv file: '+f_name)

  window_size = 200
  train_ratio = 0.4
  val_ratio   = 0.2
  test_ratio  = 0.4
  assert train_ratio + val_ratio + test_ratio == 1, 'The sum of split ratios should be 1'
  
  data_series = np.loadtxt(f_name, delimiter = ',')
  assert data_series.ndim == 2, 'The shape of input series should be (TimeFrame, Features)'
  len_train = int(data_series.shape[0] * train_ratio)
  len_val   = int(data_series.shape[0] * val_ratio)
  len_test  = int(data_series.shape[0] * test_ratio)

  windows_trn = seq_win_vectorize(data_series[:len_train,:], window_size)
  windows_val = seq_win_vectorize(data_series[len_train:len_train+len_val,:], window_size)
  windows_tst = seq_win_vectorize(data_series[len_train+len_val:,:], window_size)
  print('windows_trn size: ', windows_trn.shape)
  print('windows_val size: ', windows_trn.shape)
  print('windows_tst size: ', windows_trn.shape)

  np.save(file_name_split[0]+'_train.npy', windows_trn)
  np.save(file_name_split[0]+'_val.npy',   windows_val)
  np.save(file_name_split[0]+'_test.npy',  windows_tst)
