import numpy as np

def log_loss(pred,
             label):
    temp = 1 - pred
    temp[label-1, 0] = 1 - temp[label-1, 0]   # temp: [1-p1, 1-p2...pi, ...1-p10]
    value = -np.sum(np.log(temp))
    temp[label-1, 0] = -temp[label-1, 0]      # temp: [1-p1, 1-p2...-pi, ...1-p10]
    grad = 1 / temp
    return value, grad