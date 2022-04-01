from kaggle_environments import make
from agent import agent_vt
from lux.game import Game
def agent_dummmy(observation, configuration):
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
        actions = ['m u_1 e', 'm u_2 e']
        return actions
episodes=100
size=32
for eps in range(episodes):
    print("=== Episode {} ===".format(eps))
    env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":size, "height":size}, debug=True)
    steps = env.run([agent_vt, "simple_agent"])
    env.render(mode="ipython", width=1200, height=800)