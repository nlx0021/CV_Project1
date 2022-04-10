import gzip
import struct
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

class MINIST():
    
    def __init__(self):
        
        np.random.seed(21)
        
        train_path = osp.join('data', 'train')
        test_path = osp.join('data', 'test')
        
        train_fig_path = osp.join(train_path, 'train-images-idx3-ubyte.gz')
        train_label_path = osp.join(train_path, 'train-labels-idx1-ubyte.gz')
        test_fig_path = osp.join(test_path, 't10k-images-idx3-ubyte.gz')
        test_label_path = osp.join(test_path, 't10k-labels-idx1-ubyte.gz')
        
        # Training set and the validation set.
        with gzip.open(train_fig_path, 'rb') as figpath:
            magic, train_num, rows, cols = struct.unpack('>IIII', figpath.read(16))
            train_images = np.frombuffer(figpath.read(), dtype=np.uint8).reshape(-1, 28*28)
            
        with gzip.open(train_label_path, 'rb') as labelpath:
            magic, train_num = struct.unpack('>II', labelpath.read(8))
            train_labels = np.frombuffer(labelpath.read(), dtype=np.uint8).reshape(-1, 1) + 1
            
        idx = np.arange(train_num)
        np.random.shuffle(idx)
        new_train_images, new_train_labels = train_images[idx, :][:-10000, :], train_labels[idx][:-10000]
        valid_images, valid_labels = train_images[idx, :][-10000:, :], train_labels[idx][-10000:]
        
        self.train_set = {'fig': new_train_images, 'label': new_train_labels, 'num': train_num-10000}
        self.valid_set = {'fig': valid_images, 'label': valid_labels, 'num': 10000}
        
        # Testing set.
        with gzip.open(test_fig_path, 'rb') as figpath:
            magic, test_num, rows, cols = struct.unpack('>IIII', figpath.read(16))
            test_images = np.frombuffer(figpath.read(), dtype=np.uint8).reshape(-1, 28*28)
            
        with gzip.open(test_label_path, 'rb') as labelpath:
            magic, test_num = struct.unpack('>II', labelpath.read(8))
            test_labels = np.frombuffer(labelpath.read(), dtype=np.uint8).reshape(-1, 1) + 1
            
        self.test_set = {'fig': test_images, 'label': test_labels, 'num': test_num}
        
        # Normalization.
        mean = np.mean(self.train_set['fig'])
        var = np.var(self.train_set['fig'])
        self.mean, self.var = mean, var
        
        self.train_set['fig'] = (self.train_set['fig'] - mean) / np.sqrt(var)
        self.valid_set['fig'] = (self.valid_set['fig'] - mean) / np.sqrt(var)
        self.test_set['fig'] = (self.test_set['fig'] - mean) / np.sqrt(var)
        
        
    def get_one_data(self):
        
        num = self.train_set['num']
        idx = np.random.randint(0, num)
        
        sample = self.train_set['fig'][idx:idx+1, :].copy().T
        label = self.train_set['label'][idx, 0]
        
        return sample, label
    
    def visualize(self, one_sample):
        # Map to 256.
        one_sample = (one_sample.T * np.sqrt(self.var) + self.mean).T.astype(np.uint8)
        fig = one_sample.reshape(28, 28)
        plt.imshow(fig)
        plt.show()


if __name__ == '__main__':
    dataset = MINIST()
    sample, label = dataset.get_one_data()
    dataset.visualize(sample)
    print(label)