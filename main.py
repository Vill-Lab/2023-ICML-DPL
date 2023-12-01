'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from models import *
from optimizer import *
from utils import *



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_loss = train_loss/(batch_idx+1)
    train_acc = 100.*correct/total

    return train_loss, train_acc


# test
def test(epoch, ckpt_name):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+ ckpt_name + '.pth')
        best_acc = acc

    test_loss = test_loss/(batch_idx+1)
    test_acc = 100.*correct/total

    return test_loss, test_acc



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Implementation for DPL on CIFAR100, CIFAR-100')
    parser.add_argument('--dataset', type=str, default="CIFAR100", help="dataset ['CIFAR10'/'CIFAR100']")
    parser.add_argument('--data_dir', type=str, default="./dataset", help="dataset dirpath")
    parser.add_argument('--net', type=str, default="vgg16", help='net type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='training epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--alpha', default=0.05, type=float, help='perturbation strength')
    parser.add_argument('--G_size', default=0.05, type=float, help="the size of Guide Set ('x%' of the training set size)")
    parser.add_argument('--varepsilon', default=0.04, type=float, help="the size of Amended Training Samples ('x%' of the training set size)")
    parser.add_argument('--reg_aug', default="aug", type=str, help="the perturbation approach flag 'rep_aug' = ['rep'/'aug']")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Load network.
    print('==> Building model..')
    net = load_net(args)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint.
    if args.resume: 
        print('==> Resuming from checkpoint..')
        ckpt_path = './checkpoint/vgg16.pth'
        net, best_acc, start_epoch = load_checkpoint(net, ckpt_path)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epoch)

    DPL_iter = 0
    best_test_loss = float("inf")

    while(True):
        print('=== DPL Iteration {} Started ==='.format(DPL_iter))
        
        # Load dataset.
        print('==> Loading dataset..')
        torch.manual_seed(0)
        trainloader, testloader, ori_trainloader = load_data(args)

        # Train and test
        print('==> Starting to train and test..')
        for epoch in range(start_epoch, start_epoch + args.epoch):
            ckpt_name = args.net + "_DPL_iter_" + str(DPL_iter)
            train_loss, train_acc = train(trainloader, epoch)
            test_loss, test_acc = test(testloader, epoch, ckpt_name)
            scheduler.step()

        # Post-hoc optimization by DPL
        print('==> Performing optimization by DPL..')
        DPL_optimizer(net, trainloader, testloader, args, DPL_iter, ori_trainloader)
        print('=== DPL Iteration {} Finished ==='.format(DPL_iter))

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            DPL_iter += 1
        else:
            break

        

    