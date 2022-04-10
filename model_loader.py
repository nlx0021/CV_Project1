import pickle
from os.path import join

from network import fc_network

def model_loader(file_name):
    
    print('\nmodel loading.')
    
    # Load the network.
    layers_path = join('model', file_name+'_net.pkl')
    
    with open(layers_path, 'rb') as f:
        layers = pickle.load(f)
       
    # Get the new network. 
    new_network = fc_network(layers=layers)
    
    print('\nComplete.')
    
    return new_network