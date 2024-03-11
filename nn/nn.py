# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]], # changed Union(int, str) to Union[int,str]
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward( # COMPLETE
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]: 
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # layer linear transformed matrix
        # z^{l+1} = W^{(l)} * a^{(l)} + b^{(l)}
        Z_curr = np.matmul( W_curr, A_prev ) + b_curr 

        # activation matrix of current layer Z_curr
        # a^{(l+1)} = f(z^{(l + 1)})
        if activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif activation == "relu":
            A_curr = self._relu(Z_curr)
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]: # CHECK AFTER FINISHING BACKPROP
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # example architecture:
        # [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
        
        A = X.T # A is current hypothesis matrix
        cache = {} # cache is a dictionary of Z and A matrices

        cache["A0"] = A # stores 0th activation matrix in cache as 0th hypothesis 

        # iterates through each layer, performs _single_forward 
        for i in range(len(self.arch)):
            W_curr = self._param_dict['W' + str(i+1)]
            b_curr = self._param_dict['b' + str(i+1)]
            A_prev = A
            activation = self.arch[i]["activation"]
            print("layer:", i, "len W:", len(W_curr), "len b:", len(b_curr), activation)

            A_next, Z_next = self._single_forward(W_curr, b_curr, A_prev, activation)
            cache[f"A{i}"] = A_next 
            cache[f"Z{i}"] = Z_next
        output = A_next # outputs the final hypothesis matrix 
        
        return output, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # apply activation functions, \delta = \frac{\partial f_{activation}(Z) }{\partial Z}
        if activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        # update partial derivative matrices 
        dA_prev = np.dot( dZ_curr, A_prev) # wrt previous layer activation matrix
        dW_curr = np.dot( dZ_curr, W_curr) # wrt current layer weight matrix 
        db_curr = np.dot( Z_curr, b_curr) # wrt current layer bias matrix 

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        for i in range(len(self.arch)):
            pass

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        pass

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        pass

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:  # COMPLETE
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # \sigma(z) = \frac{1}{1 + e^{- z} } 
        return 1 /(1 + np.exp( - Z) ) # sigmoid function of Z 

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike): # COMPLETE
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # derivative of sigmoid function
        # \frac{\partial \sigma}{\partial \Z} = \sigma (Z) * (1-\sigma (Z))
        return (self._sigmoid(Z) * ( 1 - self._sigmoid(Z)) ) # * dA ?

    def _relu(self, Z: ArrayLike) -> ArrayLike: # COMPLETE
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # zero from -infinity to 0, linear 0 to +infinity
        return np.maximum(Z, np.zeros_like(Z))

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike: # COMPLETE
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # derivatie of relu 
        return (self._relu(Z) > 0 ) * dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float: # COMPLETE
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # binary cross entropy loss equation 
        # binary cross entropy loss = - \frac{1}{N} \Sum^N_{i=1} { y_i * \log( p(y_i)) + (1 - y_i) * \log(1 - p(y_i) )   }
        N = len(y_hat) # number of instances 
        
        # edit values of y_hat to prevent divide by zero error
        epsilon = 1e-15  # small constant to avoid log(0)
        for i in range(len(y_hat)):
            if y[i] == 1:
                y[i] -= epsilon
            if y[i] == 0:
                y[i] += epsilon

        # mean loss using binary cross entropy loss equation, np.log gives natrual log 
        loss = -1/N * np.sum( y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) 
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike: # COMPLETE
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # edit values of y and y_hat to prevent divide by zero error
        epsilon = 1e-15  # small constant to avoid divide by 0 error 
        for i in range(len(y)):
            if y[i] == 1:
                y[i] -= epsilon
            if y[i] == 0:
                y[i] += epsilon

        for i in range(len(y)):
            if y_hat[i] == 1:
                y_hat[i] -= epsilon
            if y_hat[i] == 0:
                y_hat[i] += epsilon
                
        # derivative of binary cross entropy (with respect to A = y_hat)
        # L(y,y_hat) = - \frac{1}{N} \Sum^N_{i=1} { y * \log(y_hat) + (1 - y) * \log(1 - y_hat )   }
        # \frac{\partial L}{\partial y_hat} =  \left( ( y * 1/y_hat ) + (1 - y) \frac{-1}{1 - y_hat} \right) (- \frac{1}{N})
        # = - \frac{1}{N} ( \frac{y}{y_hat} - \frac{1 - y}{1 -y_hat} )
        dA = - 1/self._batch_size * ( np.divide(y, y_hat) - np.divide( 1 - y, 1 - y_hat ) ) # N is batch size 
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float: # COMPLETE
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # mean squared error equation 
        # MSE(y, y_hat) = \frac{1}{N} \Sum_1^N{ (y - y_hat)^2 }
        loss = np.mean( (y - y_hat)**2 ) 
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike: # COMPLETE
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # derivative of mean squared error (with resepct to y_hat)
        # MSE(y, y_hat) = \frac{1}{N} \Sum_1^N{ (y - y_hat)^2 }
        # \frac{ \partial MSE(y,y_hat)}{\partial y_hat} = \frac{-2}{N} (y - y_hat)
        dA = -2/self._batch_size * (y - y_hat) 
        return dA