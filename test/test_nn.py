# TODO: import dependencies and write unit tests below
import pytest
from nn import (nn, io, preprocess)
import numpy as np
import sklearn.metrics
import tensorflow as tf

nn_test = nn.NeuralNetwork(nn_arch=[{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}],
                            lr = 0.001, seed = 42, batch_size= 10, epochs=3, loss_function="relu", )

def test_single_forward(): 
    # test in one dimension 
    W = np.array([2])
    b = np.array([0.5])
    a = np.array([1])
    activation_relu = "relu"
    activation_sigmoid = "sigmoid"
    sf_relu_truth = np.array([2.5]), np.array([2.5]) # z = w * a + b, 2 * 1 + 0.5 = 2.5, relu(2.5) = 2.5
    sf_sigmoid_truth = np.array([0.924141819979]), np.array([2.5]) # z = w * a + b, 2 * 1 + 0.5 = 2.5, sigmoid(2.5) = \frac{1}{1 + e^{-2.5}} = 0.924141819979

    # check if single forward pass works for known values for W, b, and a with available activation functions
    assert np.allclose( sf_relu_truth, nn_test._single_forward(W,b,a,activation_relu) ), "error in _single_forward (relu activation)"
    assert np.allclose( sf_sigmoid_truth, nn_test._single_forward(W,b,a,activation_sigmoid) ), "error in _single_forward (sigmoid activation)"
test_single_forward()

def test_forward(): 
    # input nn_arch from description of nn_arch
    X = np.ones((64,64))
    # nn_arch_test = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
    output, cache = nn_test.forward(X)
    assert np.allclose(output, np.ones_like(output)) == False, "forward pass did not change input values"
    assert 'A0' and 'A1' and 'Z1' and 'A2' and 'Z2' in cache.keys(), "forward pass did not create activation matrices"
test_forward() 

def test_single_backprop(): 
    # initalize test variables 
    W_curr = np.array([[2,1],[1,1]])
    b_curr = np.array([[0.5,1],[1,1]])
    Z_curr = np.array([[1,1],[1,1]])
    A_prev = np.array([[1,1],[1,1]])
    dA_curr = np.array([[1,1],[1,1]])
    activation_relu = "relu"
    activation_sigmoid = "sigmoid"

    # check if single backprop works on test variables
    bp_relu_test = nn_test._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_relu)
    assert np.allclose(bp_relu_test[0],  np.array([[3, 3],[2, 2]])), "single backprop did not return correct dA_prev"
    assert np.allclose(bp_relu_test[1], np.array([[2, 2],[2, 2]] )), "single backprop did not return correct dW_curr"
    assert np.allclose(bp_relu_test[2], np.array([[2],[2]])), "single backprop did not return correct db_curr"
test_single_backprop()

def test_predict():   
    # test if predict() returns values different from its input
    X = np.ones((64,64))
    pred = nn_test.predict(X)
    assert np.allclose(pred, np.ones_like(pred)) == False, "Prediction was not successful"
test_predict()

def test_binary_cross_entropy():  
    # create new y and y_hat with a known loss function value
    y_true_pytest = np.array([[0.5,0.5],[0.5,0.5]])
    y_pred_pytest = np.array([[0.5,0.5],[0.5,0.5]])
    nn_test._batch_size = 1
    loss_truth = 2.772588722239781

    # check if binary cross entropy function returns values close to true values 
    loss_pytest = nn_test._binary_cross_entropy(y= y_true_pytest, y_hat=y_pred_pytest) 
    nn_test._batch_size = len(y_true_pytest)

    assert np.isclose(loss_truth, loss_pytest, 0.000001), "binary cross entropy does not return correct result"
test_binary_cross_entropy()

def test_binary_cross_entropy_backprop(): 
    # create new y and y_hat with a known loss function value
    y_true_pytest = np.array([[0.5,0.5],[0.5,0.5]])
    y_pred_pytest = np.array([[0.5,0.5],[0.5,0.5]])
    dBCE_truth = np.zeros_like(y_true_pytest)

    # check if binary cross entropy backpropagation function returns values close to true values 
    dBCE_test = nn_test._binary_cross_entropy_backprop(y= y_true_pytest, y_hat=y_pred_pytest) 
    nn_test._batch_size = len(y_true_pytest)
    
    assert np.allclose(dBCE_truth, dBCE_test), "binary cross entropy backpropagation does not return correct result"
test_binary_cross_entropy_backprop()

def test_mean_squared_error(): 
    # test for MSE = 0
    y_test_0 = np.array([1,2,3,4,5])
    y_hat_test_0 = np.array([1,2,3,4,5])
    MSE_truth_0 = np.zeros_like(y_test_0)
    nn_test._batch_size = 5
    MSE_test_0 = nn_test._mean_squared_error(y = y_test_0, y_hat= y_hat_test_0)
    assert np.allclose( MSE_truth_0, MSE_test_0), "minimum squared error does not return zero for y = y_hat"

    # test MSE by comparing with sklearn MSE 
    y_test = np.array([[1, 0, 0],[1,1,1],[1,1,1]])
    y_hat_test = np.array([[2, 1, 4],[1,1,1],[1,1,1]])
    MSE_truth = sklearn.metrics.mean_squared_error(y_true= y_test, y_pred=y_hat_test) 
    nn_test._batch_size = 9
    MSE_test = nn_test._mean_squared_error(y = y_test, y_hat= y_hat_test)
    assert np.allclose(MSE_truth, np.sum(MSE_test) ), "minimum squared error does not return correct result"
test_mean_squared_error()


def test_mean_squared_error_backprop(): 
    # test for dMSE = 0 
    y_test_0 = np.array([1,2,3,4,5])
    y_hat_test_0 = np.array([1,2,3,4,5])
    nn_test._batch_size = 1 

    dMSE_test_0 = nn_test._mean_squared_error_backprop(y = y_test_0, y_hat= y_hat_test_0)
    dMSE_truth_0 = np.zeros_like(dMSE_test_0)
    assert np.allclose( dMSE_test_0, dMSE_truth_0 ), "MSE backpropagation does not return 0 for y = y_hat"

    # test for known dMSE
    y_true_pytest = np.array([5])
    y_pred_pytest = np.array([0.5])
    dMSE_test = nn_test._mean_squared_error_backprop(y = y_true_pytest, y_hat= y_pred_pytest)
    dMSE_truth = -9
    assert np.allclose(dMSE_test, dMSE_truth), "MSE backpropagation returns wrong result"

    # test against known dMSE for matrix 
    y_test = np.array([1, 0, 0])
    y_hat_test = np.array([2, 1, 4])
    dMSE_test_mat = nn_test._mean_squared_error_backprop(y = y_test, y_hat= y_hat_test)
    dMSE_truth_calc = np.array( [2, 2, 8] )
    assert np.allclose( dMSE_test_mat, dMSE_truth_calc), "MSE backpropagation returns wrong result"
test_mean_squared_error_backprop()

def test_sample_seqs():
    # check for correct dimensions of output, assure it works for different data types 
    t1, t2 = preprocess.sample_seqs(["yes", "no", "yes", "no", "yes", "no"], [1, 0, 1, 0, 1, 0, 1, 0])
    assert len(t1) == len(t2)
test_sample_seqs()

def test_one_hot_encode_seqs():
    # check if one hot encoding returns correct result for a known sequence and encoding
    test_AGA = preprocess.one_hot_encode_seqs("AGA")
    assert np.allclose(test_AGA, np.array([[1., 0., 0., 0.],[0., 0., 0., 1.],[1., 0., 0., 0.]]) ), "one hot encoding does not work correctly"
test_one_hot_encode_seqs()

# additional tests 
def test_relu(): 
    matrix = np.array([-1, 0, 1, 2])
    _relu_truth = [0, 0, 1, 2]
    _relu_test = nn_test._relu(matrix) 
    for i in range(len(_relu_truth)):
        assert _relu_truth[i] == _relu_test[i]
test_relu()

def test_relu_backprop(): 
    matrix = np.array([-1, 0, 1, 2])
    dA_backprop = np.array([1, 1, 1, 1])
    _relu_backprop_truth = [0, 0, 1, 1]
    _relu_test = nn_test._relu_backprop(dA_backprop, matrix) 
    for i in range(len(_relu_backprop_truth)):
        assert _relu_backprop_truth[i] == _relu_test[i]
test_relu_backprop()