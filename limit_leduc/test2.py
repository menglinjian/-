import numpy as np
from rlcard.agents.random_agent import RandomAgent
import rlcard

from agents import cfr_plus_agent
from agents import cfr_agent
from agents import DeepCFRagent5
from agents import MCCFRagent
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from nolimitleducholdem import NolimitLeducholdemEnv
from rlcard.envs.nolimitholdem import NolimitholdemEnv
import time
import multiprocessing
from multiprocessing import Pool

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, assign_task
import utils
import torch

set_global_seed(0)
chips = 15
evaluate_num = 500
Process_num = 8
env = NolimitLeducholdemEnv(allow_step_back=True,chips=chips)
eval_env = NolimitLeducholdemEnv(allow_step_back=True, chips=chips)

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
e1 = np.mean(r.computer_exploitability(agent1, evaluate_num*2, 8))
e2 = np.mean(r.computer_exploitability(agent2, evaluate_num*2, 8))
e3 = np.mean(r.computer_exploitability(agent3, evaluate_num*2, 8))
l.append([e1,e2,e3])
end = time.perf_counter()
logger.log('eposide {}:{},{},{}, test time:{}'.format(0, e1, e2, e3, end-start))

for i in range(160):
    agent1.train()
    agent2.train()
    agent3.train()
    if (i+1)%16==0:
        start = time.perf_counter()
        e1 = np.mean(r.computer_exploitability(agent1, evaluate_num*2, 8))
        e2 = np.mean(r.computer_exploitability(agent2, evaluate_num*2, 8))
        e3 = np.mean(r.computer_exploitability(agent3, evaluate_num*2, 8))
        l.append([e1,e2,e3])
        end = time.perf_counter()
        logger.log('eposide {}:{},{},{}, test time:{}'.format((i+1)/16, e1, e2, e3, end-start))


for item in l:
    print(item)


'''
agent.load()
r = utils.exploitability()
e1 = np.mean(r.computer_exploitability(agent, evaluate_num*200, 8))
print(e1)
e2 = np.mean(r.computer_exploitability(agent2, evaluate_num*200, 8))
print(e2)
e3 = np.mean(r.computer_exploitability(agent3, evaluate_num*200, 8))
print(e3)
print(e,e2,e3)
#logger.log('eposide 1:{}'.format(1,e))
'''
'''
agent.load()
agent2 = cfr_plus_agent.CFRAPlusgent(env, isAbs=True, model_path='./cfr_plus_model')
agent2.load()
agent3 = cfr_plus_agent.CFRAPlusgent(env, isAbs=False, model_path='./cfr_plus_model_noabstraction')
agent3.load()
r = utils.reward()
print(np.mean(r.computer_reward(agent, agent3, evaluate_num*200, Process_num, eval_env)))#1.8
print(np.mean(r.computer_reward(agent, agent2, evaluate_num*200, Process_num, eval_env)))#2.0
'''
'''
root_path = './DeepCFR_model_robust_attention_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR_model_robust_attention', log_path=log_path, csv_path=csv_path)                          
r = utils.exploitability()
l=[]
#e1 = np.mean(r.computer_exploitability(agent, evaluate_num*10, 8))
#print(e1)
#l.append(e1)
for i in range(40):
    agent.deepCFR(i, 1)
    e1 = np.mean(r.computer_exploitability(agent, evaluate_num*10, 8))
    logger.add_point(x=i, y=e1)
    print(e1)
    l.append(e1)
agent.save()
print(l)
logger.make_plot(save_path=figure_path+'.png')

#[3.0145999999999997, 3.9524, 4.9437, 2.9171]
'''



