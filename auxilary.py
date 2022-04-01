
import torch
import numpy as np
from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
from dataLoaders import lux_ai_data
from torch.utils.data import Dataset, DataLoader
from lux.game_objects import   Unit,CityTile
from config import unit_action_count,cart_action_count,city_action_count,resources_state_dim,num_resources,unit_state_dim,city_state_dim,player_count,saveDirectory
from callbacks import MetricsCallback
class unit_keeper(Unit,CityTile):
    def __init__(self,type,id,object,action_value=0):
        self.type=['worker','cart','city_tile'][type]
        self.id=id
        self.obj=object
        self.action_value=action_value
        self.choosen_action_index=-1

    def submission_action_interface(self,game_map):
        """
        Converts choosen action and player id to submission interface
        :param player_type:
        :param player_id:
        :param action_index:
        :return: list|player action
        """
        direction='cewns'
        if self.choosen_action_index==0:return None
        if self.type in ('worker','cart'):
            if self.choosen_action_index<5:
                return self.move(direction[self.choosen_action_index])
            elif self.choosen_action_index == 5 and self.obj.can_build(game_map):
                return self.obj.build_city()
            elif self.choosen_action_index == 6 :
                return self.pillage()
        else:
            if self.choosen_action_index==1:
                return self.obj.research()
            elif self.choosen_action_index == 2 :
                return self.obj.build_worker()
        raise ("more action than planned")
    def action_value_update(self,action_value,game_map):
        """
        changes value to -1 for actions which cannot be taken
        :param action_value:
        :param game_map:
        :return:
        """
        if self.type in ('worker','cart'):
            if not self.obj.can_act():
                action_value[1:5]=-1
        if self.type in ('worker'):
            if self.type =='worker' and not self.obj.can_build(game_map) :
                action_value[5] = -1
        return action_value


    def get_action(self,action_value,game_map,greedy_epsilon):
        action_count_mapping={'worker':range(unit_action_count),'cart':range(unit_action_count,cart_action_count),
                              'city_tile':range(cart_action_count+unit_action_count,cart_action_count+unit_action_count+city_action_count)}
        action_value = self.action_value_update(
            action_value[self.obj.pos.x, self.obj.pos.y, action_count_mapping[self.type]], game_map)
        if torch.rand(1) < greedy_epsilon:
                self.choosen_action_index = np.random.choice(len(action_value))
                self.action_value = action_value[self.choosen_action_index]
                if self.action_value==-1:self.choosen_action_index=0
                self.action_value = action_value[self.choosen_action_index]
        else:
                self.action_value, self.choosen_action_index = torch.max(action_value, dim=0)


    def get_index(self):

        if self.type=='worker': return self.choosen_action_index
        elif self.type=='cart':return self.choosen_action_index+unit_action_count
        else: return self.choosen_action_index+unit_action_count+cart_action_count


def choose_action(action_value,player_units,game_map,greedy_epsilon=0.1):
    """
    provides the action set and attched value to player units
    algo: reshape the model output>apply argmax>get the action based on argmax> attch the value to player_units
    :param y:r=tensor|output from neural net model
    :param player_units: record of all the unit player is having
    :return:[[]]|action sequence, {id:player_unt} player_units from parameter have updated value function
    """
    action_list=[]
    player_units_dict={}

    for i in player_units:
        i.get_action(action_value,game_map,greedy_epsilon)
        action=i.submission_action_interface(game_map)
        if action is not None :action_list.append(action)
        player_units_dict[i.id]=i
    return action_list,player_units_dict


def  perform_fit(model,x,y,data_loader,criterion,optimizer):
    """
    takes current model and x and y to update the model.
    Algo:create dataloaders, and pass this to catalyst runner to get ned fitted model.
    :param model:  N
    :param x:
    :param y:
    :param data_loader:
    :param criterion:
    :param optimizer:
    :return:
    """



    loaders = {
        "train": DataLoader(data_loader(state_array=x, y_values=y),
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)}
    callback=[MetricsCallback(input_key="targets", output_key="logits",
    directory=saveDirectory, model_name='lux_ai')]
    runner = SupervisedRunner(
        output_key="logits",
        input_key="image_pixels",
        target_key="targets")
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        num_epochs=1,
        verbose=True,
        logdir=f"fold0",
        callbacks=callback,
    )
    return runner.model
def input_state(game_state,observation):
    """
    Takes input from game object to form input to Q(state,action). This function is implemented using neural network
    so return output will be input to neural network.
    algo:
    1. create map for resource ,units,city state

    :param game_state:game object
    :return:list|0:torch.tensor of size game.map.width,game.map.width,19 1:dictionary{'unit':[poistin]}
    """
    w, h = game_state.map.width, game_state.map.height
    player_turn=observation.player
    step = observation['step']
    observation=observation['updates']
    #creating zero array to keep values
    r_w,r_c,r_u,u_0,u_1,ca_0,ca_1,ct_0,ct_1=tuple([torch.zeros((w,h,resources_state_dim)) for i in range(num_resources)]
    +[torch.zeros((w,h,unit_state_dim)) for i in range(player_count)]+[torch.zeros((w,h,unit_state_dim)) for i in range(player_count)]+[torch.zeros((w,h,city_state_dim)) for i in range(player_count)])
    resource={'wood':r_w,'coal':r_c,'uranium':r_u}
    #r_c[0,0,0]=98
    units=[[u_0,u_1],[ca_0,ca_1]]
    city_tile=[ct_0,ct_1]
    city_dict={}
    player_units=[]

    for upd in observation:
            splits=upd.split(" ")
            if splits[0]=='r':
                resource[splits[1]][int(splits[2]),int(splits[3]),:]=int(splits[4])#r uranium 29 31 346
            elif splits[0]=='u':
                    units[int(splits[1])][int(splits[2])][int(splits[4]),int(splits[5]),:]=torch.tensor([int(splits[6]) \
                        ,int(splits[7]),int(splits[8]),int(splits[9]) ])#'u 0 0 u_1 3 27 0 0 0 0'
                    #player_units.append(unit_keeper(type=int(splits[1]),id=int(splits[3]),position=(int(splits[4]),int(splits[5]))))
            elif splits[0]=='c':
                city_dict[splits[2]]=[int(splits[3]),int(splits[4])] #storing fuel,upkeep
            elif splits[0]=='ct':
                city_tile[int(splits[1])][int(splits[3]),int(splits[4]),:]=torch.tensor([int(splits[5])]+city_dict[splits[2]])#'ct 1 c_2 28 27 0'
                #player_units.append(
                  #  unit_keeper(type=2, id=splits[2], position=(int(splits[3]),int(splits[4]))))

    state=torch.cat((torch.dstack((r_w,r_c,r_u,u_0,u_1,ca_0,ca_1,ct_0,ct_1)).flatten(),torch.tensor([step])))
    for unit in game_state.players[player_turn].units:
        temp_unit=unit_keeper(id=unit.id, object=unit, type=unit.type)
        player_units.append(temp_unit)
    for unit in game_state.players[player_turn].cities.values():
        for sub_unit in unit.citytiles:
            temp_unit=unit_keeper(id=unit.cityid, object=sub_unit, type=2)
            player_units.append(temp_unit)

    return state,player_units

if __name__ == "__main__":
    from kaggle_environments import make

    import math, sys
    from lux.game import Game
    from models import NeuralNet
    from lux.game_map import Cell, RESOURCE_TYPES
    from lux.constants import Constants


    def agent(observation, configuration):
        global game_state

        ### Do not edit ###
        if observation["step"] == 0:
            game_state = Game()
            game_state._initialize(observation["updates"])
            game_state._update(observation["updates"][2:])
            game_state.id = observation.player
            #model = NeuralNet(**get_dict_from_class(model_param))
        else:
            game_state._update(observation["updates"])
        c=game_state
        d=input_state(observation)
        if observation.step==165:
            l=0
        actions = ['m u_1 e', 'm u_2 e']
        return actions
    env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True}, debug=True)
    steps = env.run([agent, "simple_agent"])
    env.render(mode="ipython", width=1200, height=800)
# {'remainingOverageTime': 60, 'step': 0, 'width': 32, 'height': 32, 'reward': 0, 'globalUnitIDCount': 2,
#  'globalCityIDCount': 2, 'player': 0, 'updates': ['0', '32 32', 'rp 0 0',..,'u 0 0 u_1 3 27 0 0 0 0',
# 'u 0 1 u_2 28 27 0 0 0 0', 'c 0 c_1 0 23', 'c 1 c_2 0 23', 'ct 0 c_1 3 27 0', 'ct 1 c_2 28 27 0',
# 'ccd 3 27 6', 'ccd 28 27 6', 'D_DONE']






