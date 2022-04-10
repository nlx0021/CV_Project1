import argparse
from model_loader import model_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize')
    
    parser.add_argument('--model_name', 
                        action='store',
                        type=str)
    
    args = parser.parse_args()
    
    network = model_loader(args.model_name)
    network.visualize_error()
    network.visualize_weights()