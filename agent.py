from dataLoaders import *
from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
from config import *
from auxilary import *
from funcs import get_dict_from_class, count_parameters

from losses import BCELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from catalyst import dl
import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate


DIRECTIONS = Constants.DIRECTIONS
game_state = None


def agent(observation, configuration):
    """
    Algorithm:
    initialization for step 0>collects state from game_state_object> use fn:input state to get state in tensor>execute fn:model.predict to get
    get action values>choose episilon greedy action ftn:choose_action> update last state output value with
    q_learning from action> run optimization step for 1 epoch to change papram values ftn:model.fit> load currect state
    into last state> next iteration
    :param observation:
    :param configuration:
    :return: Actions in form of string list
    """
    global game_state,model,map_size,last_q_s_a,last_player_units,criterion,optimizer

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        map_size = observation['width']

        #getting model params and model|start
        model_param_= get_dict_from_class(model_param)
        model_param_['sizes'][0]=(map_size**2)*state_dim_per_square+1
        model_param_['sizes'][-1]=(map_size**2)*action_count_per_square
        model = NeuralNet(**model_param_)

        optimizer = optim.SGD(model.parameters(), lr=lr)
        # getting model params and model|stop
        # last state initialization
        last_q_s_a = None
        last_player_units=None
    else:
        game_state._update(observation["updates"])
    actions = []
    state_tensor,player_units = input_state(game_state,observation)  # state tensor:1d player_units:unit_kkeper object
    model_output = model(state_tensor)  # output:1d
    q_s_a= model_output.reshape((map_size, map_size, action_count_per_square)) # reshaping to position*actions
    actions,player_unit_dict=choose_action(action_value=q_s_a,player_units=player_units ,game_map=game_state.map)
    if last_q_s_a is not None:#update the Q learning matrix
        reward = observation['reward']
        for u in  last_player_units:
            if u.choosen_action_index !=-1:
                q_last_state=last_q_s_a[u.obj.pos.x,u.obj.pos.y,u.get_index()]
                if u.id in player_unit_dict.keys(): q_state=player_unit_dict[u.id].action_value
                else: q_state=extinct_value
                temporal_difference=reward+gamma*q_state-q_last_state
                q_last_changed = q_last_state + step_size*temporal_difference
                last_q_s_a[u.obj.pos.x,u.obj.pos.y, u.get_index()]=q_last_changed

        #model fitting prep and run
        model = perform_fit(model=model,x=state_tensor,y=last_q_s_a.flatten(),data_loader=data_loader,criterion=criterion,optimizer=optimizer)
    last_player_units=player_units
    last_q_s_a = q_s_a
    print(actions)
    return actions


if __name__ == "__main__":
    import torch.nn.functional as F
    import torch

    c = NeuralNet(sizes=[5, 12, 12], act_funcs=[F.relu_ for i in range(3)])
    d = c(torch.tensor([i for i in range(5)], dtype=torch.float32))
    print(d.shape)
