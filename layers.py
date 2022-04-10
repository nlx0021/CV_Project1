import numpy as np
from scipy.signal import convolve2d


class fc_layer():
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 sigma,
                 bias,
                 drop):
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        np.random.seed(21)        
        self.init_weights(sigma)
        
        self.bias = bias
        self.drop = drop
        self.momentum_W = np.zeros((out_dim, in_dim))
        self.momentum_b = np.zeros((out_dim, 1))
        
        # For visualize.
        self.loss = []
        self.error = []
        self.iter = []
    
    
    def init_weights(self,
                     sigma):
        # Weights.
        self.W = np.random.normal(0, sigma, size=(self.out_dim, self.in_dim))
        # Bias.
        self.b = np.zeros((self.out_dim, 1))
        
    def forward(self,
                x,
                train=True):
        if train:
            rnd = np.random.random(x.shape)    
            mask = rnd <= self.drop
            x = x * mask
        else:
            x = x * self.drop
        self.x = x
        if self.bias:
            return np.dot(self.W, x) + np.tile(self.b, x.shape[1])
        else:  
            return np.dot(self.W, x)
    
    def backward(self,
                 grad):
        x = self.x
        self.grad_W = np.dot(grad, x.T)
        if self.bias:
            self.grad_b = np.sum(grad, axis=1).reshape(-1, 1)
        return np.dot(self.W.T, grad)
    
    def step(self,
             lr,
             lamb=0):
        
        self.W = self.W - lr * (self.grad_W + lamb * self.W)
        if self.bias:
            self.b = self.b - lr * self.grad_b
            
        

class conv_layer():
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernal_size,
                 sigma,
                 drop):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal_size = kernal_size
        self.drop = drop
        
        np.random.seed(21)
        self.init_weights(sigma)
        
    def init_weights(self,
                     sigma):
        self.W_list = []
        k = self.kernal_size
        c = self.in_channels
        out_channels = self.out_channels
        for _ in range(out_channels):
            self.W_list.append(np.random.normal(0, sigma, (k, k, c)))
            
    def forward(self,
                X,
                train=True):
        # Padding.
        pad_size = (self.kernal_size - 1) // 2
        X = np.pad(X, ((pad_size, pad_size), (pad_size, pad_size), (0,0)))
        
        # Drop.
        if train:
            rnd = np.random.random(X.shape)    
            mask = rnd <= self.drop
            X = X * mask            
        else:
            X = X * self.drop
        
        self.X = X
        # Conv for every kernal.
        output_list = []    # Contain each layer of the output.
        for W in self.W_list:
            one_output = 0  # Compute one layer of output.
            for i in range(self.in_channels):
                one_output = one_output + convolve2d(X[:, :, i], self.flip(W[:, :, i]), 'valid')
            output_list.append(one_output)
        output = np.stack(output_list, axis=2)
        return output
    
    def backward(self,
                 grad):
        # Compute grad for W.
        self.grad_W_list = []
        X = self.X
        k = self.kernal_size
        h, w, _ = X.shape
        for i in range(self.out_channels):
            one_grad_W_list = []  # Contain each grad of W in W_list.
            for j in range(self.in_channels):
                one_grad_W_list.append(convolve2d(X[:, :, j], self.flip(grad[:, :, i]), 'valid'))
            self.grad_W_list.append(np.stack(one_grad_W_list, axis=2))
        
        # Compute grad for X.
        output = 0
        for i in range(self.out_channels):
            one_grad_X_list = []  # Contain each layer of one partial grad.
            temp = np.pad(grad[:, :, i], ((k-1,k-1), (k-1,k-1)))
            for j in range(self.in_channels):
                one_grad_X_list.append(convolve2d(temp, self.W_list[i][:, :, j], 'valid'))
            output = output + np.stack(one_grad_X_list, axis=2)
        pad_size = (self.kernal_size - 1) // 2
        output = output[pad_size:h-pad_size, pad_size:w-pad_size, :]   # Depad.
        
        return output
    
    def step(self,
             lr,
             momentum=True,
             lamb=0):
        for i in range(self.out_channels):
            self.W_list[i] = self.W_list[i] - lr * self.grad_W_list[i]
               
    def conv(self,
             X,
             W):
        # X size: [h, w, c] 
        # W size: [k, k, c]
        c = W.shape[2]
        result = 0
        for i in range(c):
            result = result + convolve2d(X[:, :, i], self.flip(W[:, :, i]), 'valid')
        return result
         
    def flip(self,
             array):
        # Flip a 2D array.
        new_arr = array.reshape(array.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(array.shape)
        return new_arr


class activate_layer():
    
    def __init__(self,
                 type):
        self.type = type
        
    def forward(self,
                x,
                *_):
        self.x = x
        if self.type == 'sigmoid':
            self.output = 1 / (np.exp(-x) + 1)
        elif self.type == 'tanh':
            self.output = (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
        elif self.type == 'relu':
            self.output = x * (x > 0)
        return self.output
    
    def backward(self,
                 grad):
        a = self.output
        if self.type == 'sigmoid':
            return a * (1-a) * grad
        elif self.type == 'tanh':
            return (1 - a**2) * grad
        elif self.type == 'relu':
            return (a > 0) * grad
        
    def step(self,
             *_):
        pass
        
    
class softmax_layer():
    
    def __init__(self):
        pass
    
    def forward(self,
                x,
                *_):
        self.x = x
        exp_x = np.exp(x)
        self.output = exp_x / np.sum(exp_x, axis=0)
        return self.output
    
    def backward(self,
                 grad):
        a = self.output
        A = -np.dot(a, a.T)
        A = A + np.diag(a.reshape((-1,)))
        return np.dot(A.T, grad)
    
    def step(self,
             *_):
        pass
    

class flatten_layer():
    
    def __init__(self,
                 in_size):
        
        self.in_size = in_size
        
    def forward(self,
                X,
                *_):
        return X.reshape(-1, 1)
    
    def backward(self,
                 grad):
        return grad.reshape(-1, self.in_size, 1)
    
    def step(self,
             *_):
        pass


class pooling_layer():
    
    def __init__(self, stride):
        self.stride = stride
        
    def forward(self, X):
        pass