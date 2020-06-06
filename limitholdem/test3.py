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
from agents import DeepCFRagent3
import math
import rlcard
#from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, assign_task
import utils

set_global_seed(100)
chips = 100
evaluate_num = 500
Process_num = 8
env = LimitholdemEnv(allow_step_back=True,chips=chips)

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
#agent0.save()
agent0.load()
#agent0.model_path='./DeepCFR+_model2'
#agent0.load()

print(">>Hold'em")

env.init_game()

while True:
    if env.is_over():
        print(env.get_payoffs())
        break
    current_player = env.get_player_id()
    obs,legal_actions,state = agent0.get_state(current_player)
    if current_player==0:
        agent0.env.print_state(0,legal_actions)
        action = input('>> You choose action (integer): ') 
        agent0.env.step(int(action))
    else:
        action = agent0.eval_step(state)
        agent0.env.step(action)