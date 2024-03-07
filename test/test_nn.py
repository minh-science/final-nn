# TODO: import dependencies and write unit tests below
from nn import nn
import numpy as np

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass


nn_test = nn.NeuralNetwork(nn_arch=[{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}],
                            lr = 0.001, seed = 42, batch_size= 10, epochs=3, loss_function="relu", )
# additional tests 
def test_relu():
    matrix = np.array([-1, 0, 1, 2])
    _relu_truth = [0, 0, 1, 2]
    _relu_test = nn_test._relu(matrix) 
    for i in range(len(_relu_truth)):
        assert _relu_truth[i] == _relu_test[i]
test_relu()

