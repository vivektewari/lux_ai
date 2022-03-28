from kaggle_environments import make
from agent import agent

env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True}, debug=True)
steps = env.run([agent, "simple_agent"])
env.render(mode="ipython", width=1200, height=800)