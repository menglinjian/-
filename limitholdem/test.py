import numpy as np
from rlcard.agents.random_agent import RandomAgent
import rlcard
from agents import cfr_plus_agent2
from agents import MCCFRagent2
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from limitholdem import LimitholdemEnv
import time
import multiprocessing
from multiprocessing import Pool
from agents import DeepCFRagent
import math
import rlcard
#from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, assign_task
import utils

set_global_seed(0)
chips = 100
evaluate_num = 500
Process_num = 8
env = LimitholdemEnv(allow_step_back=True,chips=chips)
eval_env = LimitholdemEnv(allow_step_back=True, chips=chips)
'''
agent0 = DeepCFRagent3.DeepCFRagent(env,
                    epochs = 100, # NN epochs
                    mini_batches = 15,
                    CFR_num = 1,
                    tra_num = 10, 
                    startPolicy = 0,
                    robustSample = False, 
                    sample_num = 5,
                    sample_rate = 1,
                    is_training = True, 
                    regret_memory_size = 400,
                    policy_memory_size = 600, 
                    batch_size = 32,
                    model_path='./DeepCFR+_model')
'''
agent0 = DeepCFRagent.DeepCFRagent(env,
                    epochs = 1, # NN epochs
                    mini_batches = 1,
                    CFR_num = 1,
                    tra_num = 10, 
                    robustSample = False, 
                    sample_num = 5,
                    is_training = True, 
                    memory_size = 600, 
                    batch_size = 32, 
                    model_path='./DeepCFR_model')
#agent0 = MCCFRagent2.MCCFRagent(env, isAbs=False, CFR_num=1, tra_num=10)
#agent1 = cfr_plus_agent2.CFRAPlusgent(env, isAbs=True, CFR_num=1, tra_num=2)
#agent2 = cfr_plus_agent2.CFRAPlusgent(env, isAbs=False, CFR_num=1, tra_num=2)
agent3 = RandomAgent(action_num=env.action_num)
l=[]

from rlcard.utils.logger import Logger
root_path = './model_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR+_model', log_path=log_path, csv_path=csv_path) 

r = utils.reward()
'''
start = time.perf_counter()
e1 = np.mean(r.computer_reward(agent0, agent2, evaluate_num*20, Process_num, eval_env))
e2 = np.mean(r.computer_reward(agent1, agent2, evaluate_num*20, Process_num, eval_env))
end = time.perf_counter()
logger.log('eposide {}:{:.5f},{:.5f} test time:{}'.format(0, e1, e2, end-start))
'''

for i in range(100):
    start = time.perf_counter()
    agent0.deepCFR(i, 8)
    #agent1.train(i,8)#20*8*1*1
    #agent2.train(i,8)
    e1 = np.mean(r.computer_reward(agent0, agent3, evaluate_num*50, Process_num, eval_env))
    e2 = 1#np.mean(r.computer_reward(agent1, agent3, evaluate_num*50, Process_num, eval_env))
    e3 = 1#np.mean(r.computer_reward(agent2, agent3, evaluate_num*50, Process_num, eval_env))
    end = time.perf_counter()
    l.append([e1,e2,e3])
    logger.log('eposide {}:{:.5f},{:.5f},{:.5f} test time:{}'.format(i+1, e1, e2, e3, end-start))
    #print(e1, e2, e3)
agent0.save()
#agent1.save()
for item in l:
    print(item)
'''
agent2 = cfr_plus_agent.CFRAPlusgent(env, isAbs=True, model_path='./cfr_plus_model')
agent2.load()
agent3 = cfr_plus_agent.CFRAPlusgent(env, isAbs=False, model_path='./cfr_plus_model_noabstraction')
agent3.load()

from rlcard.utils.logger import Logger
root_path = './DeepCFR+_model3_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR+_model', log_path=log_path, csv_path=csv_path)    

#r = utils.reward()
r = utils.exploitability()
l=[]
start = time.perf_counter()
e = np.mean(r.computer_exploitability(agent, evaluate_num, 8))
l.append(e)
end = time.perf_counter()
logger.log('eposide {}:{}, test time:{}'.format(0, e, end-start))

e1 = np.mean(r.computer_reward(agent, agent3, evaluate_num*200, Process_num, eval_env))
e2 = np.mean(r.computer_reward(agent, agent2, evaluate_num*200, Process_num, eval_env))
l.append([e1,e2])
logger.log('eposide {}:{},{}'.format(0, e1,e2))

for i in range(10):
    agent.deepCFR(i, 8)
    start = time.perf_counter()
    e = np.mean(r.computer_exploitability(agent, 1, 8))
    l.append(e)
    end = time.perf_counter()
    logger.log('eposide {}:{}, test time:{}'.format(i, e, end-start))

agent.save()

for item in l:
    print(item)

r = utils.exploitability()
e = np.mean(r.computer_exploitability(agent, evaluate_num*50, 8))
logger.log('last eposide exploitability:{}'.format(e))
'''
