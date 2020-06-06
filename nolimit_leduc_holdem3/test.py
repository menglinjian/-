import numpy as np
from rlcard.agents.random_agent import RandomAgent
import rlcard
'''
from agents import cfr_agent_mp 
from agents import cfr_agent_original_mp  
from agents import bias_agent_mp 
from agents import cfr_agent 
from agents import cfr_agent_original 
from agents import bias_agent
''' 
from agents import cfr_plus_agent
'''
from agents import cfr_plus_agent_mp
from agents import cfr_plus_agent_subgame
'''
from agents import DeepCFRagent
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
chips = 10
evaluate_num = 500
Process_num = 8
env = NolimitLeducholdemEnv(allow_step_back=True,chips=chips)
eval_env = NolimitLeducholdemEnv(allow_step_back=True, chips=chips)
agent = DeepCFRagent.DeepCFRagent(env,#
                                epochs = 100, # NN epochs
                                mini_batches = 15, #NN train batches of a epochs#100
                                CFR_num = 2,
                                tra_num = 2, #20
                                robustSample = True, 
                                startPolicy = 0,
                                sample_num = 5,
                                sample_rate = 1,
                                is_training = True,  
                                regret_memory_size = 300,#300, 
                                policy_memory_size = 400,#400, 
                                batch_size = 32,
                                model_path='./DeepCFR+_model4') 
'''
agent2 = cfr_plus_agent.CFRAPlusgent(env, isAbs=True, model_path='./cfr_plus_model')
agent2.load()
agent3 = cfr_plus_agent.CFRAPlusgent(env, isAbs=False, model_path='./cfr_plus_model_noabstraction')
agent3.load()
'''
agent.model_path='./DeepCFR+_model4temp3'
#agent.save()
#agent.load()
test_state = {'legal_actions': [0, 1, 6], 'my_chips': 6, 'public_cards': ['HJ'], 'all_chips': [8, 6], 'hand': ['HK'], 'current_player': 1, 'round': 1, 'action_history': np.array([[5., 3., 7., 0., 0., 0., 0., 0.],
       [1., 1., 5., 0., 0., 0., 0., 0.]])}

from rlcard.utils.logger import Logger
root_path = './DeepCFR+_model4_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR+_model', log_path=log_path, csv_path=csv_path)    

l=[]

r = utils.exploitability()
start = time.perf_counter()
agent.currentP = False
e1 = np.mean(r.computer_exploitability(agent, evaluate_num*5, 8))#-0.1549
agent.currentP = True
e2 = np.mean(r.computer_exploitability(agent, evaluate_num*5, 8))#1.9433
#2.7411858974358974,2.2479967948717947
l.append([e1,e2])
end = time.perf_counter()
logger.log('eposide {}:{},{}, test time:{}'.format(0, e1, e2, end-start))

for i in range(25):
    agent.deepCFR(i, 8)
    start = time.perf_counter()
    agent.currentP = False
    e1 = np.mean(r.computer_exploitability(agent, evaluate_num*5, 8))
    agent.currentP = True
    e2 = np.mean(r.computer_exploitability(agent, evaluate_num*5, 8))
    l.append([e1,e2])
    end = time.perf_counter()
    logger.log('eposide {}:{},{}, test time:{}'.format(i+1, e1, e2, end-start))
    '''
    logger.log('eposide {}:{}, test time:{}\nregrets:{}\naverage:{}\n{}\ncurrent:{}'.format(i+1, e, end-start
                                                            ,agent.get_regrets(test_state)[0:7]
                                                            ,agent.get_policy(test_state, range(0,7), isEval=True)[0:7]
                                                            ,agent.get_policy(test_state, [0, 1, 6], isEval=True)[0:7]
                                                            ,agent.get_policy(test_state, [0, 1, 6], isEval=False)[0:7]))
    '''

agent.model_path='./DeepCFR+_model4temp7'
agent.save()

for item in l:
    print(item)
agent.currentP = False
r = utils.exploitability()
start = time.perf_counter()
e = np.mean(r.computer_exploitability(agent, evaluate_num*20, 8))
end = time.perf_counter()
print(e)
logger.log('last eposide:{},test time:{}'.format(e, end-start))
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



