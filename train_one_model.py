import argparse
import numpy as np
from layers import fc_layer, activate_layer, softmax_layer
from network import fc_network
from data_loader import MINIST
from model_loader import model_loader
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_one_model')
    
    parser.add_argument('--model_name', 
                        action='store',
                        type=str)
    parser.add_argument('--lr',
                        action='store',
                        type=float)
    parser.add_argument('--max_iter',
                        action='store',
                        type=int)
    parser.add_argument('--lamb',
                        action='store',
                        type=float)
    parser.add_argument('--neural_num',
                        action='store',
                        type=int)
    
    args = parser.parse_args()
    
    neural_num = args.neural_num
    max_iter = args.max_iter
    lr = args.lr
    lamb = args.lamb
    model_name = args.model_name
    
    net_work = fc_network(
        layers = {
            'fc1': fc_layer(in_dim=784, out_dim=neural_num, sigma=np.sqrt(2/(784+neural_num)), bias=True, drop=1),
            'activ1': activate_layer(type='tanh'),
            'fc2': fc_layer(in_dim=neural_num, out_dim=10, sigma=np.sqrt(2/(10+neural_num)), bias=True, drop=1),
            'softmax': softmax_layer()
            }
    )       
    
    dataset = MINIST()
    
    error = net_work.train(max_iter=max_iter,
                        lr=lr,
                        dataset=dataset,
                        loss='log_loss',
                        lamb=lamb,
                        visualize_loss=False,
                        visualize_weights=False)
    
    net_work.model_save(model_name)