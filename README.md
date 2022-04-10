# CV_Project1
The first project of cv.

## train——参数查找
若想进行训练的参数查找，请直接运行train.py文件。

```python train.py```
  
运行此文件后将会在以下参数中进行参数查找：1. 学习率，包括0.002、0.001和0.0005； 2. 正则化系数，包括0.01和0.001； 3. 中间层神经元个数，包括50、100和300个。
  
得到的最佳模型会自动保存在model文件夹下，名字为"My_best_model_net.pkl"
## train——训练自己的模型
若想自定义参数进行训练，请运行train_one_model.py文件并传入参数。例如：
  
```python train_one_model.py --model_name My_model --lr 0.0001 --max_iter 100000 --lamb 0.001 --neural_num 100```
  
（lr为学习率，max_iter为最大迭代数，lamb为正则化惩罚系数，neural_num为中间层的神经元个数)
  
模型将被保存在model文件夹中。注意，若您在传入参数时模型名字为"xxxx"，那么最终模型保存的名字为"xxxx_net.pkl"，即会有额外的后缀。
## test——测试自己的模型
请运行test.py文件并传入模型的名字。例如：
  
```python test.py --model_name My_model```

模型在测试集上的精度将会输出在终端。注意，在传入您的模型名字时，请不要把模型的文件名输入，即不要带上最后_net.pkl的后缀。例如：对于文件名为"xxxx_net.pkl"的模型，那么若想读入此模型，请在传入参数时只穿入"xxxx"作为模型名字。
## visualize——可视化自己的模型
请运行visualize.py文件并传入模型的名字。例如：
  
```python visualize.py --model_name My_model```
  
模型的loss曲线，精度曲线以及各层参数的可视化将被显示。注意模型名字的传入（同上）。

## Pretrained model
我们训练好的模型在————————中可以下载。下载后，请将其保存到model文件夹下。
