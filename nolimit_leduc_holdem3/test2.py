import numpy as np
from rlcard.agents.random_agent import RandomAgent
import rlcard
from agents import cfr_plus_agent2
from agents import cfr_plus_agent
from agents import cfr_agent
from agents import MCCFRagent
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from nolimitleducholdem import NolimitLeducholdemEnv
import time
import multiprocessing
from multiprocessing import Pool
from agents import DeepCFRagent3

import rlcard
#from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, assign_task
import utils

set_global_seed(0)
chips = 10
evaluate_num = 500
Process_num = 8
env = NolimitLeducholdemEnv(allow_step_back=True,chips=chips)
eval_env = NolimitLeducholdemEnv(allow_step_back=True, chips=chips)


agent = MCCFRagent.MCCFRagent(env, isAbs=False)
from rlcard.utils.logger import Logger
root_path = './model_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR+_model', log_path=log_path, csv_path=csv_path)

l=[]
r = utils.exploitability()
start = time.perf_counter()
e1 = np.mean(r.computer_exploitability(agent, evaluate_num*5, 8))
l.append([e1])
end = time.perf_counter()
logger.log('eposide {}:{} test time:{}'.format(0, e1, end-start))

for i in range(800):
    agent.train()
    if (i+1)%32==0:
        start = time.perf_counter()
        e1 = np.mean(r.computer_exploitability(agent, evaluate_num*5, 8))
        l.append([e1])
        end = time.perf_counter()
        logger.log('eposide {}:{} test time:{}'.format((i+1)/32, e1, end-start))
for item in l:
    print(item)

    
'''
agent1 = agent = cfr_agent.CFRAgent(env, isAbs=False)
agent2 = cfr_plus_agent.CFRAPlusgent(env, isAbs=False)
agent3 = MCCFRagent.MCCFRagent(env, isAbs=False)

from rlcard.utils.logger import Logger
root_path = './model_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR+_model', log_path=log_path, csv_path=csv_path)    

l=[]

r = utils.exploitability()
start = time.perf_counter()
e1 = np.mean(r.computer_exploitability(agent1, evaluate_num*5, 8))
e2 = np.mean(r.computer_exploitability(agent2, evaluate_num*5, 8))
e3 = np.mean(r.computer_exploitability(agent3, evaluate_num*5, 8))
l.append([e1,e2,e3])
end = time.perf_counter()
logger.log('eposide {}:{},{},{}, test time:{}'.format(0, e1, e2, e3, end-start))

for i in range(800):
    agent1.train()
    agent2.train()
    agent3.train()
    if (i+1)%32==0:
        start = time.perf_counter()
        e1 = np.mean(r.computer_exploitability(agent1, evaluate_num*5, 8))
        e2 = np.mean(r.computer_exploitability(agent2, evaluate_num*5, 8))
        e3 = np.mean(r.computer_exploitability(agent3, evaluate_num*5, 8))
        l.append([e1,e2,e3])
        end = time.perf_counter()
        logger.log('eposide {}:{},{},{}, test time:{}'.format((i+1)/32, e1, e2, e3, end-start))


for item in l:
    print(item)
'''