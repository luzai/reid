configs_str= \
'''
arch: resnet50
batch_size: 60
branch_dim: 64
branchs: 8
combine_trainval: true
data_dir: /home/xinglu/.torch/data
dataset: cuhk03
decay_epoch: 100
dist_metric: euclidean
dropout: 0
epochs: 180
evaluate: false
features: 1024
freeze: false
gpu: [3]
height: 256
logs_dir: ../work/logs.res.1.sgd
loss: triplet
lr: 0.1
margin: 0.5
mode: hard
normalize: false
num_classes: 128
num_instances: 4
optimizer: sgd
print_freq: 5
resume: ''
seed: 1
split: 0
start_save: 180
steps: [100, 150]
use_global: false
weight_decay: 0.0005
width: 128
workers: 32

arch: resnet50
dataset: cuhk03
# resume: '../examples/logs.ori/model_best.pth.tar'
resume: ''
evaluate: False
optimizer: sgd  
normalize: False
dropout: 0 
features: 1024
num_classes: 128 
lr: 0.1
steps: [100,150,]
epochs: 180
logs_dir: logs.vis
batch_size: 60
gpu: [3,]
'''