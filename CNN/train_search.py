import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  #Network初始化一个8层网络
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),#优化器更新的参数，这里更新的是w
      args.learning_rate,#初始值是0.025，采用的是余弦退火调度更新学习率，每个epoch的学习率都不一样
      momentum=args.momentum,#momentum = 0.9
      weight_decay=args.weight_decay)#正则化参数3e-4

  train_transform, valid_transform = utils._data_transforms_cifar10(args)#图像变换处理
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  #training data前一半用于training set
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)
  #training data后一半用于validation set
  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(#出自Paper'SGDR:Stochastic Gradient DescentWarm Restarts'
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)#采用余弦退火调度设置各组参数组的学习率
  #创建用于更新alpha的architect
  architect = Architect(model, args)
  #经历50个epoch后搜索完成
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]#得到本次迭代的学习率lr
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()#对应论文2.4 选出来权重值大的两个前驱节点，并把最后的结果存下来，格式为Genotype(normal=[(op,i),..],normal_concat=[],reduce=[],reduce_concat=[])
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))#输出normal cell的alpha矩阵
    print(F.softmax(model.alphas_reduce, dim=-1))#输出reduction cell的alpha矩阵

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()#用来保存loss的值
  top1 = utils.AvgrageMeter()#前1预测正确的概率
  top5 = utils.AvgrageMeter()#前5预测正确的概率

  for step, (input, target) in enumerate(train_queue):#每个step取出一个batch，batchsize=64，就是256个数据对
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()#ariable默认是不需要被求导的，即requires_grad属性默认为False，可以搭建计算图
    target = Variable(target, requires_grad=False).cuda()#async=True

    # get a random minibatch from the search queue with replacement
    #更新alpha是用validation set 进行更新的，所以我们每次都从valid_queue拿出一个batch传入architect.py
    input_search, target_search = next(iter(valid_queue))#用于架构参数更新一个batch，使用iter返回的是一个迭代器，可以使用next进行访问
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()#async=True
    
    #对α进行更新，对应伪代码的第一步，也就是用公式6
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)#调用architect里的step函数
    #对w进行更新，对应伪代码的第二步
    optimizer.zero_grad()#清除之前学到的梯度参数
    logits = model(input)
    loss = criterion(logits, target)#预测值logits和真实值target的loss

    loss.backward()#进行反向传播，计算梯度
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)#梯度裁剪
    optimizer.step()#应用梯度

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()#async=True

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

