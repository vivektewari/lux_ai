import numpy as np



def input_state(game_state):
    """
    Takes input from game object to form input to Q(state,action). This function is implemented using neural network
    so return output will be input to neural network.
    algo:
    1. create map for resource ,units,city state

    :param game_state:game object
    :return:np.array of size game.map.width,game.map.width,19
    """
    w, h = game_state['width'], game_state['height']
    step =game_state['step']
    #creating zero array to keep values
    r_w,r_c,r_u,u_0,u_1,ca_0,ca_1,ct_0,ct_1=tuple([np.zeros((w,h,1)) for i in range(3)]
    +[np.zeros((w,h,4)) for i in range(4)]+[np.zeros((w,h,3)) for i in range(2)])
    resource={'wood':r_w,'coal':r_c,'uranium':r_u}
    #r_c[0,0,0]=98
    units=[[u_0,u_1],[ca_0,ca_1]]
    city_tile=[ct_0,ct_1]
    city_dict={}

    for upd in game_state['updates']:
            splits=upd.split(" ")
            if splits[0]=='r':
                resource[splits[1]][int(splits[2]),int(splits[3]),:]=int(splits[4])#r uranium 29 31 346
            elif splits[0]=='u':
                    units[int(splits[1])][int(splits[2])][int(splits[4]),int(splits[5]),:]=int(splits[6]) \
                        ,int(splits[7]),int(splits[8]),int(splits[9])
            elif splits[0]=='c':
                city_dict[splits[2]]=[int(splits[3]),int(splits[4])] #storing fuel,upkeep
            elif splits[0]=='ct':
                city_tile[int(splits[1])][int(splits[3]),int(splits[4]),:]=tuple([int(splits[5])]+city_dict[splits[2]])#'ct 1 c_2 28 27 0'

    state=np.append(np.dstack((r_w,r_c,r_u,u_0,u_1,ca_0,ca_1,ct_0,ct_1)).flatten(),step)

    return state

if __name__ == "__main__":
    from kaggle_environments import make
    from agent import agent
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
            model = NeuralNet(**get_dict_from_class(model_param))
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







