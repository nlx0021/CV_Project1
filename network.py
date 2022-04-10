import numpy as np
import pickle
import matplotlib.pyplot as plt
from layers import fc_layer, activate_layer
from loss import log_loss
from data_loader import MINIST
from math import pi
from copy import deepcopy
from os.path import join

class fc_network():
    
    def __init__(self, layers):
        self.layers = layers
        self.stop_flag = 0

    def train(self,
              max_iter,
              lr,
              dataset,
              loss='log_loss',
              lamb=0,
              visualize_loss=False,
              visualize_weights=False):
        for iter in range(max_iter):
            
            # Get data.
            x_in, x_label = dataset.get_one_data() 
        
            # Forward.
            for layer in self.layers.values():
                x_in = layer.forward(x_in)
            
            # Compute loss.
            yhat = x_in
            f, g_in = log_loss(yhat, x_label)
            
            # Backward.
            for layer in list(self.layers.values())[::-1]:
                g_in = layer.backward(g_in)
            
            # Step.
            lr_schedule = lr * (1 + np.cos(iter*pi / max_iter)) / 2
            for layer in self.layers.values():
                layer.step(lr_schedule, lamb)
            
            # Validation.    
            if iter % 10000 == 0 or (iter < 10000 and iter % 1000 == 0):
                error, loss = self.predict(dataset, 'valid')
                list(self.layers.values())[0].error.append(error)
                list(self.layers.values())[0].iter.append(iter)
                list(self.layers.values())[0].loss.append(loss)
                print('\niters: %5d' % iter, sep=', ')
                print('valid error rate: %.5f' % error)
                print('average loss: %.5f' % loss)
        
        # Test (Valid).
        error, _ = self.predict(dataset, 'valid')
        print('\nvalid error rate: %.5f' % error)
        
        # Loss lines.
        if visualize_loss:
            self.visualize_error()
            
        # Weights.
        if visualize_weights:
            self.visualize_weights()
        
        return error
    
    
    def predict(self, dataset, type):
        if type == 'valid':
            data = dataset.valid_set['fig']
            label = dataset.valid_set['label']
            num = dataset.valid_set['num']
        elif type == 'test':
            data = dataset.test_set['fig']
            label = dataset.test_set['label']
            num = dataset.test_set['num']
        
        x_in = data.T
        # Forward.
        for layer in self.layers.values():
            x_in = layer.forward(x_in, False)  # train=False
        
        yhat = x_in
        
        # Compute the error.
        pred = (np.argmax(yhat, axis=0).T + 1).reshape((-1, 1))
        error = np.sum((pred - label) != 0) / label.shape[0]
        
        # Compute the loss.
        loss = 0
        for i in range(num):
            loss += log_loss(yhat[:, i:i+1], label[i, 0])[0]
        loss = loss / num

        return error, loss


    def model_save(self, file_name):
        
        print('\nModel saving.')
        
        layers = deepcopy(self.layers)
           
        with open(join('model', file_name+'_net.pkl'), 'wb') as f:
            f.truncate()
            pickle.dump(layers, f)
            
        print('\nComplete.')
        
    def visualize_error(self):
        layer_first = list(self.layers.values())[0]
        iters = layer_first.iter
        errors = layer_first.error
        losses = layer_first.loss
        
        plt.plot(iters, errors, label='Error in valid', linewidth=3, color='r',
                 marker='o', markerfacecolor='blue', markersize=5)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Error rate')
        plt.legend()
        plt.show()
        
        plt.plot(iters, losses, label='Loss in valid', linewidth=3, color='r',
                 marker='o', markerfacecolor='blue', markersize=5)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()        
        
    def visualize_weights(self):
        layer_1 = self.layers['fc1']
        layer_2 = self.layers['fc2']
        
        # Visualize for the first layer.
        idx = np.arange(layer_1.out_dim)
        np.random.shuffle(idx)
        idx = idx[:9]
        count = 0 
        for i in idx:
            weights = layer_1.W[i, :].copy()
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))   # Normalize.
            weights = (weights * 255).astype(np.uint8)
            fig = weights.reshape((-1, 28))
            
            plt.subplot(3, 3, count+1)
            plt.imshow(fig)
            plt.xticks([])
            plt.yticks([])
            count += 1 
        
        plt.show()
        
        # Visualize for the second layer.
        idx = np.arange(layer_2.out_dim)
        np.random.shuffle(idx)
        idx = idx[:9]
        count = 0         
        for i in idx:
            weights = layer_2.W[i:i+1, :].copy()
            weights = np.dot(weights, layer_1.W)
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))   # Normalize.
            weights = (weights * 255).astype(np.uint8)
            fig = weights.reshape((-1, 28))
            
            plt.subplot(3, 3, count+1)
            plt.imshow(fig)
            plt.xticks([])
            plt.yticks([])
            count += 1 
        plt.show()                   