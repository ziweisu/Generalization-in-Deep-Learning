from tqdm import tqdm
import sys
import torchvision
import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from utils import AverageMeter, save_checkpoint, accuracy
from loss import BeliefMatchingLoss

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batchsize', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weightdecay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--path1', default='', type=str, 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--path2', default='', type=str, 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default='1,2', type=str)
parser.add_argument('--coeff', default=1e-2, type=float, help='Coefficient to KL term in BM loss. Set -1 to use CrossEntropy Loss')
parser.add_argument('--prior', default=1.0, type=float, help='Dirichlet prior parameter')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num-eval', dest='num_eval', default=100, type=int, help='Evaluation count for MC dropout')

parser.add_argument('--output_name', default='-1', type=str)


best_prec1 = 0
test_error_best = -1

def main():
    # load test data
    global args, best_prec1
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if(args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 10
    elif(args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 100
        
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
       
    # model define
    model1 = resnet.ResNet18(num_classes)
    model2 = resnet.ResNet18(num_classes)
        
    model1 = torch.nn.DataParallel(model1)
    model2 = torch.nn.DataParallel(model2)
    
    model1.cuda()
    model2.cuda()
    
    # model load
    modelname = 'checkpoint_resnet18-softmax-cifar10.th'
    checkpoint = torch.load(os.path.join(args.path1, modelname))
    checkpoint2 = torch.load(os.path.join(args.path2, modelname))
    
    model1.load_state_dict(checkpoint['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    
     # eval test
    acc, disagree = validate(test_loader, model1, model2)
    
    # 
    np.save(os.path.join('results', args.output_name), 
            np.array([(100. - acc)*1e-2, disagree]))
    

def compute_disagreement(pred1, pred2):
    return torch.mean((torch.argmax(pred1, -1) != torch.argmax(pred2, -1)).float())
   
    
def validate(val_loader, model1, model2):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    disagreement = AverageMeter()
    mi_meter = AverageMeter()

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output1 = model1(input_var)    
            output2 = model2(input_var)    
                               
            # measure accuracy and disagree
            prec1 = accuracy(output1.data, target)[0]
            disagr = compute_disagreement(output1, output2)
            top1.update(prec1.item(), input.size(0))
            disagreement.update(disagr.item(), input.size(0))   

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}, Disagreement: {disagreement.avg:.3f}'
          .format(top1=top1, disagreement=disagreement))

    return top1.avg, disagreement.avg    


if __name__ == '__main__':
    main()
