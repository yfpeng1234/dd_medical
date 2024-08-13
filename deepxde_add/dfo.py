#seems that for low dimension (small IPC), random initialization is better, for high dimension, real initialization is better

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
import random
import nevergrad as ng

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#args
inner_loop=1000
dd_num=40
budget=400

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pde(x, y):
    '''
    x is a N*2 array, where x[:,[0]] is the x coordinate, x[:,[1]] is the boundary condition
    y is a N*1 array, where y is the solution of the PDE
    '''
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return -dy_xx - np.pi ** 2 * torch.sin(np.pi * x[:,[0]])

def boundary(x, on_boundary):
    return on_boundary

def BC_Prior(bc_num):
    return np.random.normal(0,1,bc_num)

def get_dataset(bc_num):
    '''
    get a unified dataset consist of data from different bc
    return: train_x_all,train_y_all,test_x_all,test_y_all
    shape: N1*2, N1*1, N2*2, N2*1
    '''
    #prepare dataset
    train_x_all=[]
    train_y_all=[]
    test_x_all=[]
    test_y_all=[]
    priors=BC_Prior(bc_num)
    for prior in priors:
        geom = dde.geometry.Interval(-1, 1)
        def func_basic(x):
            return np.sin(np.pi * x)+prior
        def pde_basic(x, y):
            dy_xx = dde.grad.hessian(y, x)
            return -dy_xx - np.pi ** 2 * torch.sin(np.pi * x)
        bc= dde.icbc.DirichletBC(geom, func_basic, boundary)
        data = dde.data.PDE(geom, pde_basic, bc, 16, 2, solution=func_basic, num_test=100)
        train_data=data.train_next_batch()
        train_x,train_y=train_data[0],train_data[1]
        train_x_aux=np.ones_like(train_x)*prior
        train_x=np.concatenate((train_x,train_x_aux),axis=-1)
        test_data=data.test()
        test_x,test_y=test_data[0],test_data[1]
        test_x_aux=np.ones_like(test_x)*prior
        test_x=np.concatenate((test_x,test_x_aux),axis=-1)
        train_x_all.append(train_x)
        train_y_all.append(train_y)
        test_x_all.append(test_x)
        test_y_all.append(test_y)
    train_x_all=np.concatenate(train_x_all,axis=0)
    train_y_all=np.concatenate(train_y_all,axis=0)
    test_x_all=np.concatenate(test_x_all,axis=0)
    test_y_all=np.concatenate(test_y_all,axis=0)
    training_data={'x':train_x_all,'y':train_y_all}
    testing_data={'x':test_x_all,'y':test_y_all}
    return training_data,testing_data

set_random_seed(44)
training_data,testing_data=get_dataset(100)
layer_size = [2] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"

def objective(x):
    x=x.astype(np.float32)
    net = dde.nn.FNN(layer_size, activation, initializer)
    model=dde.Model_General(net,pde,{'x':x[:,[0,1]],'y':x[:,[-1]]},training_data,testing_data)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    model.train(iterations=inner_loop)
    return model._test('train')

choose_idx=np.random.choice(training_data['x'].shape[0],dd_num,replace=False)
start_point=np.concatenate((training_data['x'][choose_idx],training_data['y'][choose_idx]),axis=-1).astype(np.float32)
instrum = ng.p.Instrumentation(
    ng.p.Array(init=start_point)
)
optimizer = ng.optimizers.NgIohTuned(parametrization=instrum, budget=budget)
recommendation = optimizer.minimize(objective)

dd_data=recommendation.value[0][0].astype(np.float32)

net=dde.maps.FNN(layer_size, activation, initializer)
model=dde.Model_General(net,pde,{'x':dd_data[:,[0,1]],'y':dd_data[:,[-1]]},training_data,testing_data)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
model.train(iterations=inner_loop)
result=model._test('eval')

print("inner_loop:",inner_loop)
print("dd_num:",dd_num)
print("budget:",budget)
print("result:",result)