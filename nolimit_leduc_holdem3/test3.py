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
from agents import DeepCFRagent3
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

set_global_seed(1)
chips = 10
evaluate_num = 10
Process_num = 8

env = NolimitLeducholdemEnv(allow_step_back=True,chips=chips)
eval_env = NolimitLeducholdemEnv(allow_step_back=True, chips=chips)

agent0 = DeepCFRagent3.DeepCFRagent(env)
agent0.model_path='./DeepCFR+_model4temp6'
agent0.load()

print(">> Leduc Hold'em")

env.init_game()

while True:
    if env.is_over():
        print(env.get_payoffs())
        break
    current_player = env.get_player_id()
    obs,legal_actions,state = agent0.get_state(current_player)
    if current_player==0:
        agent0.env.print_state(0)
        action = input('>> You choose action (integer): ') 
        agent0.env.step(int(action))
    else:
        action = agent0.eval_step(state)
        agent0.env.step(action)

'''
from rlcard.agents.random_agent import RandomAgent
# Set up agents
agent1 = RandomAgent(action_num=env.action_num)
r = utils.reward()
print(np.mean(r.computer_reward(agent0, agent1, evaluate_num, Process_num, eval_env)))
'''

'''
agent = DeepCFRagent3.DeepCFRagent(env,#tra_num = 5, Process_num=8, sample_rate=0.2, memory_size=40659
                                epochs = 100, # NN epochs
                                mini_batches = 20, #NN train batches of a epochs#100
                                CFR_num = 4,
                                tra_num = 5, #20
                                robustSample = True, 
                                sample_num = 5,
                                sample_rate = 0.2,
                                is_training = True, 
                                memory_size = 80000, 
                                batch_size = 2048, 
                                model_path='./DeepCFR+_model2') 


agent2 = DeepCFRagent3.DeepCFRagent(env,#tra_num = 5, Process_num=8, sample_rate=0.2, memory_size=40659
                                epochs = 100, # NN epochs
                                mini_batches = 20, #NN train batches of a epochs#100
                                CFR_num = 4,
                                tra_num = 5, #20
                                robustSample = True, 
                                sample_num = 5,
                                sample_rate = 0.2,
                                is_training = True, 
                                memory_size = 80000, 
                                batch_size = 2048, 
                                model_path='./temp') 

agent3 = DeepCFRagent3.DeepCFRagent(env,#tra_num = 5, Process_num=8, sample_rate=0.2, memory_size=40659
                                epochs = 100, # NN epochs
                                mini_batches = 20, #NN train batches of a epochs#100
                                CFR_num = 4,
                                tra_num = 5, #20
                                robustSample = True, 
                                sample_num = 5,
                                sample_rate = 0.2,
                                is_training = True, 
                                memory_size = 80000, 
                                batch_size = 2048, 
                                model_path='./DeepCFR+_model') 
agent4 = DeepCFRagent3.DeepCFRagent(env,#tra_num = 5, Process_num=8, sample_rate=0.2, memory_size=40659
                                epochs = 100, # NN epochs
                                mini_batches = 20, #NN train batches of a epochs#100
                                CFR_num = 4,
                                tra_num = 5, #20
                                robustSample = True, 
                                sample_num = 5,
                                sample_rate = 0.2,
                                is_training = True, 
                                memory_size = 80000, 
                                batch_size = 2048, 
                                model_path='./DeepCFR+_model3') 
#cureent， average
agent.load()#-0.81764，-0.70075
#agent2.load()#0.4495，-0.7933000000000001
#agent3.load()#3.4343
agent4.load()#-1.90225
r = utils.reward()

print(np.mean(r.computer_reward(agent, agent2, evaluate_num*100, Process_num, eval_env)))#3.01326
print(np.mean(r.computer_reward(agent2, agent, evaluate_num*100, Process_num, eval_env)))#-3.8885199999999998
print(np.mean(r.computer_reward(agent, agent3, evaluate_num*100, Process_num, eval_env)))#-0.1674
print(np.mean(r.computer_reward(agent3, agent, evaluate_num*100, Process_num, eval_env)))#-0.25502
print(np.mean(r.computer_reward(agent2, agent3, evaluate_num*100, Process_num, eval_env)))#-3.1847
print(np.mean(r.computer_reward(agent3, agent2, evaluate_num*100, Process_num, eval_env)))#2.9695
3.5654200000000005
-3.9432
-2.0674
1.69074
2.4919000000000002
-3.2039400000000002

print(np.mean(r.computer_reward(agent, agent4, evaluate_num*100, Process_num, eval_env)))#-0.026639999999999997
print(np.mean(r.computer_reward(agent4, agent, evaluate_num*100, Process_num, eval_env)))#-0.25862
print(np.mean(r.computer_reward(agent2, agent4, evaluate_num*100, Process_num, eval_env)))#-0.026639999999999997
print(np.mean(r.computer_reward(agent4, agent2, evaluate_num*100, Process_num, eval_env)))#-0.25862
print(np.mean(r.computer_reward(agent3, agent4, evaluate_num*100, Process_num, eval_env)))#-0.026639999999999997
print(np.mean(r.computer_reward(agent4, agent3, evaluate_num*100, Process_num, eval_env)))#-0.25862

'''
