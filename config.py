import time
from pathlib import Path
import random
import os
from dataLoaders import *
from losses import *
from models import *
import torch.optim as optim

from param_options import *

#fit related functions

#model = NeuralNet
data_loader = lux_ai_data
model_param = param_1
lr=0.01
loss_func = l2Loss()
criterion = loss_func
pretrained=None


#q learning
gamma = 0.95
step_size = 0.1


# game params
greedy_episilon = 1
extinct_value=-100
unit_action_count = 7 #0 is no action
cart_action_count=6 #0 is no action
city_action_count = 3 #0 th is no action
unit_state_dim = 4
cart_state_dim = 4
city_state_dim = 3
resources_state_dim = 1
num_resources=3
player_count = 2
action_count_per_square=unit_action_count+cart_action_count+city_action_count
state_dim_per_square = player_count*(unit_state_dim+cart_state_dim+city_state_dim)+resources_state_dim*num_resources



#directory used
root ='/home/pooja/PycharmProjects/lux_ai/'
dataCreated = root+'/data/dataCreated/'
raw_data=root+ '/data/'
image_loc =dataCreated +'/mixed/'
blank_loc =dataCreated + '/auxilary/'
saveDirectory = root + '/outputs/weights/'
device = 'cpu'
config_id = str(os.getcwd()).split()[-1]
startTime = time.time()
random.seed(23)

