import numpy as np
from layers import fc_layer, activate_layer, softmax_layer
from network import fc_network
from data_loader import MINIST
from model_loader import model_loader
from os.path import join

if __name__ == '__main__':
    
    dataset = MINIST()
    
    # Search for the hyperparameters.
    lr_set = [2e-3, 1e-3, 5e-4]
    lamb_set = [.01, .001]
    neural_num_set = [100, 200, 300]
    
    lowest_error = 1.0
    best_paras = None
    
    with open(join('model', 'search_process.txt'), 'w') as f:
        f.truncate()
    
    for lr in lr_set:
        for lamb in lamb_set:
            for neural_num in neural_num_set:
                
                net_work = fc_network(
                    layers = {
                        'fc1': fc_layer(in_dim=784, out_dim=neural_num, sigma=np.sqrt(2/(784+neural_num)), bias=True, drop=1),
                        'activ1': activate_layer(type='tanh'),
                        'fc2': fc_layer(in_dim=neural_num, out_dim=10, sigma=np.sqrt(2/(10+neural_num)), bias=True, drop=1),
                        'softmax': softmax_layer()
                        }
                )                
                    
                print('\n')
                print('Parameters: ' + str((lr, lamb, neural_num)))
                
                error = net_work.train(max_iter=300000,
                                       lr=lr,
                                       dataset=dataset,
                                       loss='log_loss',
                                       lamb=lamb,
                                       visualize_loss=False,
                                       visualize_weights=False)
                print('\n')
                print('One model finish. The error of validation is %5f' % error)
                
                with open(join('model', 'search_process.txt'), 'a') as f:
                    f.write(str((lr, lamb, neural_num))+'\n')
                    f.write(str(error)+'\n')
                
                if error < lowest_error:
                    lowest_error = error
                    best_paras = (lr, lamb, neural_num) 
                    print('\n')
                    print('Parameters update.')
                    net_work.model_save('My_best_model')
                
    print(best_paras)
    with open(join('model', 'best_paras.txt'), 'w') as f:
        f.truncate()
        f.write(str(best_paras))
        f.write(str(lowest_error))
    
    
