# TODO: import dependencies and write unit tests below
from nn import nn
import numpy as np
import sklearn.metrics

import tensorflow as tf

nn_test = nn.NeuralNetwork(nn_arch=[{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}],
                            lr = 0.001, seed = 42, batch_size= 10, epochs=3, loss_function="relu", )

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():  # COMPLETE
    # create new y and y_hat with a known loss function value
    y_true_pytest = np.array([0.5])
    y_pred_pytest = np.array([0.5])
    loss_truth = 0.69314718056

    # check if binary cross entropy function returns values close to true values 
    loss_pytest = nn_test._binary_cross_entropy(y= y_true_pytest, y_hat=y_pred_pytest) 
    nn_test._batch_size = len(y_true_pytest)

    assert np.isclose(loss_truth, loss_pytest, 0.000001), "binary cross entropy does not return correct result"
test_binary_cross_entropy()

def test_binary_cross_entropy_backprop(): # COMPLETE
    # create new y and y_hat with a known loss function value
    y_true_pytest = np.array([5])
    y_pred_pytest = np.array([0.5])
    dBCE_truth = -18

    # check if binary cross entropy backpropagation function returns values close to true values 
    dBCE_test = nn_test._binary_cross_entropy_backprop(y= y_true_pytest, y_hat=y_pred_pytest) 
    nn_test._batch_size = len(y_true_pytest)
    
    assert dBCE_truth == dBCE_test, "binary cross entropy backpropagation does not return correct result"
test_binary_cross_entropy_backprop()

def test_mean_squared_error(): # COMPLETE
    y_test_0 = np.array([1,2,3,4,5])
    y_hat_test_0 = np.array([1,2,3,4,5])
    
    MSE_truth_0 = 0
    MSE_test_0 = nn_test._mean_squared_error(y = y_test_0, y_hat= y_hat_test_0)
    assert MSE_truth_0 == MSE_test_0, "minimum squared error does not return zero for y = y_hat"

    y_test = np.array([1, 0, 0])
    y_hat_test = np.array([2, 1, 4])
    MSE_truth = sklearn.metrics.mean_squared_error(y_true= y_test, y_pred=y_hat_test)
    MSE_test = nn_test._mean_squared_error(y = y_test, y_hat= y_hat_test)
    assert MSE_truth == MSE_test, "minimum squared error does not return correct result"
test_mean_squared_error()


def test_mean_squared_error_backprop(): # finish this 
    y_test_0 = np.array([1,2,3,4,5])
    y_hat_test_0 = np.array([1,2,3,4,5])

    dMSE_truth = 0 
    dMSE_test = nn_test._mean_squared_error_backprop(y = y_test_0, y_hat= y_hat_test_0)
    print(dMSE_test)
    

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass


# additional tests 
def test_relu(): # COMPLETE
    matrix = np.array([-1, 0, 1, 2])
    _relu_truth = [0, 0, 1, 2]
    _relu_test = nn_test._relu(matrix) 
    for i in range(len(_relu_truth)):
        assert _relu_truth[i] == _relu_test[i]
test_relu()

def test_relu_backprop(): # COMPLETE
    matrix = np.array([-1, 0, 1, 2])
    dA_backprop = np.array([1, 1, 1, 1])
    _relu_backprop_truth = [0, 0, 1, 1]
    _relu_test = nn_test._relu_backprop(dA_backprop, matrix) 
    for i in range(len(_relu_backprop_truth)):
        assert _relu_backprop_truth[i] == _relu_test[i]
test_relu_backprop()