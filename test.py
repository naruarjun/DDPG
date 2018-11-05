import sys
import os
# sys.path.append(os.getcwd()+"/PointEnvironment")
from PointEnvironment.Environment import PointEnvironment
from PointEnvironment.Agent import Agent
from PointEnvironment.Pose import Pose
visualOptions = { 'tailLength' : 4, # tail of agent's trajectory
                  'speedup' : 10, # realTime/simTime
                  'bounds': [-10,10,-10,10]# bounds of the environment [xmin, xmax, ymin, ymax]
                  }
options = { 'num_iterations' : 50, # no. of steps environment takes for one step call
            'dt' : 0.01, # dt in in kinametic update
            'visualize' : False, # show visualization?
            'visualOptions' : visualOptions # visual options
            }

env = PointEnvironment(**options)
agent = Agent(0)
env.addAgent(agent)
# env.startVisualizer()
for i in range(1000):
  env.step({0: [2, 0.5]})
  print(env.agents[0].pose)