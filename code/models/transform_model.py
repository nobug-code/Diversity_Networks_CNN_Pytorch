import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
import numpy as np
from models.vgg import VGG
import utils.k_dpp as k_dpp

def tensor_to_cpu(weight, bias):
       
    weight = weight.data.cpu().numpy()
    bias = bias.data.cpu().numpy()

    return weight,bias

def np_to_tensor(weight,bias):
       
    weight = torch.from_numpy(weight).cuda()
    weight = Parameter(weight)
    bias = torch.from_numpy(bias).cuda()
    bias = Parameter(bias)

    return weight,bias

def tensor_to_parameter(weights):

    total_weight = []
    for weight in weights:
        weight = Parameter(weight)
        total_weight.append(weight)
    return total_weight

def cpu_to_tensor(weights):

    total_weight = []
    for weight in weights:
        weight = torch.FloatTensor(weight).cuda()
        weight = Parameter(weight)
        total_weight.append(weight)

    return total_weight

def get_model(args, number):

    model = None
    if(args.model == 'dpp_vgg16'):
        model = VGG('VGG16',0)
        model = nn.DataParallel(model)
        model.cuda()
        save_name = 'checkpoint_model_' + str(number) +'.tar'
        model_load = torch.load('save_vgg16_cifar10_best/' + save_name)
        model.load_state_dict(model_load['state_dict'])

    if model is None:
        print("Model is None")
        raise TypeError

    return model

def push_weight(model, before_model, weight, k_number_list):

    cfs = [0,3,7,10,14,17,20,24,27,30,34,37]
    for i in range (len(cfs)):
        model.module.features[cfs[i]].weight = before_model.module.features[cfs[i]].weight
        model.module.features[cfs[i]].bias = before_model.module.features[cfs[i]].bias
        model.module.features[cfs[i]].weight.requires_grad = False
        model.module.features[cfs[i]].bias.requires_grad = False
    model.module.features[40].weight = weight[0] 
    
    return model

def get_models_kernel(total_models):

    total_models_kernel = []
    total_models_bias = []
    for model in total_models:
        model_weight = []
        model_bias = []
        for kernel in model.modules():
            if isinstance(kernel, nn.Conv2d):
                weight, bias = tensor_to_cpu(kernel.weight, kernel.bias)
                model_weight.append(weight)
                model_bias.append(bias)
        total_models_kernel.append(model_weight)
        total_models_bias.append(model_bias)
    return total_models_kernel, total_models_bias

def k_dpp_kernel(weight, k_number):

    x = weight
    x_ = x
    x_size = x_.size()
    out_channel = x_size[0]
    in_channel = x_size[1]
    f_shape = x_size[2]
    filter_list = np.zeros((out_channel, in_channel, f_shape, f_shape))
    sim_mat = []
    for i in range(out_channel):
        filter_i = transforms.Lambda(lambda y : y[i:(i+1),:,:,:])(x)
        filter_i = filter_i.data.cpu().numpy()
        filter_list[i] = filter_i
    for i in range(out_channel):
        filter_i = filter_list[i]
        sim_mat_i = []
        for j in range(out_channel):
            d_ij = np.exp(-np.sum(pow((filter_i - filter_list[j]), 2), axis=(0,1,2)))
            sim_mat_i.append(d_ij)
        sim_mat.append(sim_mat_i)
    
    sim_mat = np.asarray(sim_mat)
    sim_mat = torch.from_numpy(sim_mat).cuda().float()
    evals, evecs = torch.symeig(sim_mat, eigenvectors=True)
    evals = evals.cpu().numpy()
    evecs = evecs.cpu().numpy()
    values = k_dpp.k_sample(k_number, evals, evecs)
    values = np.reshape(values, len(values))
    values = values.astype(int)
    weight = weight[values]

    return weight
    
def integrated_kernel(args):

    models = args.total_models.split(',')
    k_number_list = args.k_number.split(',')
    for i in range(len(k_number_list)):
        k_number_list[i] = int(k_number_list[i])

    #models is number of [1,2,3]
    total_models = []
    for model in models:
        temp = get_model(args, model)
        total_models.append(temp)

    #Get Kernel weight and Bias
    weights, bias = get_models_kernel(total_models)

    #Concat total kernel weight and Bias
    all_weights = []
    for i in range(len(weights[0])):
        allModel = weights[0][i]
        for j in range(1, len(weights)):
            allModel = np.concatenate([allModel, weights[j][i]], 0)
        all_weights.append(allModel)

    concat_weights = cpu_to_tensor(all_weights)
    #k_dpp_kernel, k_number_list
    k_weights = []
    k_bias = []
    len_number = len(concat_weights) - 1
    for i in range(len(k_number_list)):
        t_weight = k_dpp_kernel(concat_weights[len_number], k_number_list[i])
        len_number -= 1
        k_weights.append(t_weight)
    k_weights = tensor_to_parameter(k_weights)
    #model push
    if(args.model == 'dpp_vgg16'):
        model = VGG('VGG16_2', k_number_list[0])
        model = nn.DataParallel(model).cuda()
        #best_model
        b_model = get_model(args, args.best_model_num)

    model = push_weight(model, b_model, k_weights, len(k_number_list))
    return model
