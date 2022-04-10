import argparse
from data_loader import MINIST
from model_loader import model_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize')
    
    parser.add_argument('--model_name', 
                        action='store',
                        type=str)
    
    args = parser.parse_args()
    
    dataset = MINIST()
    
    network = model_loader(args.model_name)
    error, _ = network.predict(dataset, 'test')
    print('\ntest error rate: %.5f' % error)
    