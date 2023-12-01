#! /usr/bin/env python3
import torch
import time
import datetime
import numpy as np
import copy
import logging
import torch.nn as nn
import argparse


from torchvision import utils as vutils
from pathlib import Path
from torch.autograd import grad
from utils import *



def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads



def calculate_hessian_inverse(h_inputs, h_targets, net, dataloader, gpu=-1, 
            damp=0.01, scale=25.0, recursion_depth=50, H_calc_method="linalg"):
    """
    Calculates the Inverse Hessian Vector Product.

    Arguments:
        h_inputs: torch tensor, sample data points, such as train images
        h_targets: torch tensor, contains all training data labels
        net: torch NN, net used to evaluate the dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        H_inverse: list of torch tensors, the Inverse Hessian Vector Product."""
   
    net.eval()
    if gpu >= 0:
        h_inputs, h_targets = h_inputs.cuda(), h_targets.cuda()
    outputs = net(h_inputs)
    losses = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(outputs), h_targets, weight=None, reduction='mean')
    params = [ p for p in net.parameters() if p.requires_grad ]
    v = list(grad(losses, params, create_graph=True))
    H = v.copy()
    H_estimate = v.copy()

    # Compute sum of gradients from net parameters to loss

    if H_calc_method == "linalg":
        for input, label in dataloader:
            if gpu >= 0:
                input, label = input.cuda(), label.cuda()

            output = net(input)
            loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output), label, weight=None, reduction='mean')
            H = torch.cat(hvp(loss, params, v))
        H /= len(dataloader.dataset)
        H_inverse = torch.linalg.inv(H)

    else:
        for i in range(recursion_depth):
            for input, label in dataloader:
                if gpu >= 0:
                    input, label = input.cuda(), label.cuda()
                output = net(input)
                loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output), label, weight=None, reduction='mean')
                hv = hvp(loss, params, H_estimate)

                # Recursively caclulate h_estimate
                H_estimate = [
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, H_estimate, hv)]
                break
        H_inverse = H_estimate

    return H_inverse



def calculate_G_loss(net, args, trainloader):
    """Calculates the empirical risk on the guide set size. 
    The optimal size of G is 0.04*(the size of training set) in our manuscript.
    
    Returns:
        running_loss / G_size: Derivative of the loss on G.
    """

    G_size = args.G_size * len(trainloader.dataset)

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        running_loss += loss.item()

        if i % G_size == G_size - 1:    
            print('val_set_avg_loss: %.3f' %
                    (running_loss / G_size))
            return running_loss / G_size



def calculate_perturbation_single(net, trainloader, testloader, args, ori_trainloader,
            pt_sample,pt_label, gpu, recursion_depth):
            
    G_loss = calculate_G_loss(net, args, ori_trainloader)
    H_inverse = calculate_hessian_inverse(trainloader, net, testloader,gpu=gpu, recursion_depth=recursion_depth)
    
    # Calculating the second order derivative 
    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(pt_sample), pt_label, weight=None, reduction='mean')
    params = [ p for p in net.parameters() if p.requires_grad ]
    grad_params = grad(loss, params, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(grad_params, pt_sample):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    grad_sec = grad(elemwise_products, pt_sample, create_graph=True) + grad(params, pt_sample, create_graph=True)
    
    perturbation_value = G_loss * H_inverse * grad_sec
    if args.rep_aug == "rep":
        perturbation_value = args.alpha * perturbation_value
        perturbation_sort = args.alpha * torch.linalg.norm(perturbation_value)
    elif args.rep_aug == "aug":
        
        perturbation_value = args.alpha * perturbation_value
        perturbation_sort = torch.linalg.norm(G_loss * H_inverse * grad_params) + args.alpha * torch.linalg.norm(perturbation_value)
    else:
        raise(ValueError("Invalid value for the perturbation approach flag 'rep_aug' = ['rep'/'aug']."))
    
    perturbation_value = perturbation_value.numpy()

    return perturbation_value, perturbation_sort
                          


def calculate_perturbation(net, trainloader, testloader, args, DPL_iter, ori_trainloader):
    """Calculates the training sample perturbation and saves as json."""

    outdir = "./DPL_json/" + args.net + "_DPL_iter_" + str(DPL_iter) + ".json"
    perturbation_dict_list = []
    train_dataset_size = len(trainloader.dataset)


    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        perturbation_value, perturbation_sort = calculate_perturbation_single(
            net, trainloader, testloader, args, ori_trainloader, inputs,labels, gpu=1, recursion_depth=5000)

        perturbation_dict = {}
        perturbation_dict['label'] = labels
        perturbation_dict['num_in_dataset'] = i
        perturbation_dict['perturbation_value'] = perturbation_value.tolist()
        perturbation_dict['perturbation_sort'] = perturbation_sort.tolist()
        perturbation_dict_list.append(perturbation_dict)

        save_json(perturbation_dict_list, outdir)
        display_progress("Test samples processed: ", i, train_dataset_size)
        get_match_case() 

    perturbation_dict_list = sorted(perturbation_dict_list, key = lambda perturbation_dict_list: perturbation_dict_list['perturbation_sort'],reverse=True)

    return perturbation_dict_list



def get_perturbation_value(perturbation_dict_list, amend_sample_id):
    for key in perturbation_dict_list:
        if key['num_in_dataset'] == amend_sample_id:
            return key




def DPL_optimizer(net, trainloader, testloader, args, DPL_iter, ori_trainloader):
    perturbation_dict_list = calculate_perturbation(net, trainloader, testloader, args, DPL_iter, ori_trainloader)

    train_dataset_size = len(trainloader.dataset)

    # get the id list of training samples needed to be amended
    amend_sample_id = []
    for i in range(train_dataset_size):
        amend_sample_id.append(perturbation_dict_list[i]['num_in_dataset'])

    # amend the training samples
    amend_count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        for k in range(args.batch_size):
            if (batch_idx * args.batch_size + k) in amend_sample_id:
                amend_count += 1

                sample = get_perturbation_value(perturbation_dict_list, (batch_idx * args.batch_size + k))
                perturbation_value = torch.from_numpy(sample['perturbation_value'])
                amend_sample = torch.add(inputs[k], perturbation_value, out=None)

                amended_img_path = args.data_dir+"/"+args.dataset+"/"+str(train_dataset_size+amend_count)+"_"+sample['label']+".png"
                vutils.save_image(amend_sample, amended_img_path, normalize=False)
                write_csv(args, [amended_img_path, sample['label']])
