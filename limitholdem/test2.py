import numpy as np
from rlcard.agents.random_agent import RandomAgent
import rlcard
from agents import cfr_plus_agent2
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from nolimitholdem import NolimitholdemEnv
import time
import multiprocessing
from multiprocessing import Pool
from agents import DeepCFRagent3

import rlcard
#from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, assign_task
import utils

set_global_seed(0)
chips = 15
evaluate_num = 500
Process_num = 8
env = NolimitholdemEnv(allow_step_back=True,chips=chips)
eval_env = NolimitholdemEnv(allow_step_back=True, chips=chips)

agent2 = cfr_plus_agent2.CFRAPlusgent(env, isAbs=True, CFR_num=1, tra_num=100, model_path='./cfr_plus_model')
agent3 = cfr_plus_agent2.CFRAPlusgent(env, isAbs=False, CFR_num=1, tra_num=100, model_path='./cfr_plus_model_noabstraction')

from rlcard.utils.logger import Logger
root_path = './/cfr_plus_model_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='DeepCFR+_model', log_path=log_path, csv_path=csv_path)    

#r = utils.reward()
r = utils.exploitability()
l=[]
start = time.perf_counter()
e1 = np.mean(r.computer_exploitability(agent2, evaluate_num, 8))
e2 = np.mean(r.computer_exploitability(agent3, evaluate_num, 8))
l.append([e1,e2])
end = time.perf_counter()
logger.log('eposide {}:{},{} test time:{}'.format(0, e1, e2, end-start))

for i in range(10):
    agent2.train(i, 8)
    agent3.train(i, 8)
    start = time.perf_counter()
    e1 = np.mean(r.computer_exploitability(agent2, evaluate_num, 8))
    e2 = np.mean(r.computer_exploitability(agent3, evaluate_num, 8))
    l.append([e1,e2])
    end = time.perf_counter()
    logger.log('eposide {}:{},{} test time:{}'.format(i, e1, e2, end-start))


agent2.save()
agent3.save()
for item in l:
    print(item)
