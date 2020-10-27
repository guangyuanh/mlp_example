### Multi-layer Perceptron (MLP) Classification

Example PyTorch code to train and test a mlp model. The code is tested under Python 3.8.3 and PyTorch 1.5.1.

#### Window Generation

MLP models require data windows for training and testing. You can generate data windows from a time series by running
```shell
python series2window.py [path_to_csv]
```

Note that the input time series format should be (TimeFrame, Features). The default window size is 100 and you can change it in the series2window.py.

#### Model Training

Before training, you need to properly combine a few data window files (with you own code) to generate the training set (train_mlp_data.npy) and the validation set (val_mlp_data.npy). You also need to generate a training label file (train_mlp_label.npy) and a validation label file (val_mlp_label.npy) in which you have a vector of labels to mark each window as 0 or 1. To do training, you can run
```shell
python main-train.py
```

The default size of a single hidden layer is 100. The default learning rate is 0.001. The default batch size is 4. You can change them in main-train.py. The trained model for PC is saved as testing-PC-[num_hidden].pt which will be read by the testing program. The same model for experiments on smartphone is saved as testing-smartphone-[num_hidden].pt. A log of training is saved as training.out.

#### Model Testing

Before testing, you need to properly combine a few data window files (with you own code) to generate the testing set (test_mlp_data.npy) and a golden answer file (test_mlp_label.npy) in which you label each testing window as 0 or 1. By running the python code, the testing-PC-[num_hidden].pt file you generated in training is read and testing is performed. The program will print out the classification result. By default, the code reads the model with a single hidden layer of 100 nodes.
```shell
python main-test.py
```
