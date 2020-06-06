import multiprocessing
from multiprocessing import Pool
import numpy as np
from rlcard.utils.utils import set_global_seed
import random

class reward():
    def __init__(self):
        pass

    def computer_reward(self, agent1, agent2, evaluate_num, process_num, eval_env):
        eval_env.set_agents([agent1, agent2])
        try:
            agent1.set_evalenv(eval_env)
        except:
            pass
        try:
            agent2.set_evalenv(eval_env)
        except:
            pass

        from multiprocessing import Manager
        multiprocessing.freeze_support()
        u = []
        p = Pool(process_num)
        for i in range(process_num):
            u.append(p.apply_async(self.traverse, args=(agent1, agent2, int(evaluate_num/process_num), eval_env)))

        for i in range(0, len(u)):
            u[i] = u[i].get()
   
        return u

    def traverse(self, agent1, agent2, evaluate_num, eval_env):
        reward = []
        set_global_seed(random.randint(0,100))
        for eval_episode in range(evaluate_num):
            try:
                agent1.oppoCV = None
            except:
                pass
            try:
                agent2.oppoCV = None
            except:
                pass
            his, payoffs = eval_env.run(is_training=False)
            reward.append(payoffs[0])
        #print(reward)
        return np.mean(reward)

class exploitability():
    def __init__(self):
        pass

    def computer_exploitability(self, agent, evaluate_num, process_num):
        from multiprocessing import Manager
        self.agent = agent
        multiprocessing.freeze_support()
        u = []
        p = Pool(process_num)
        for i in range(process_num):
            u.append(p.apply_async(self.traverse, args=(int(evaluate_num/process_num),)))

        p.close()
        p.join()

        for i in range(0, len(u)):
            u[i] = u[i].get()

        return np.mean(u)

    def traverse(self, evaluate_num):
        reward = []
        set_global_seed(random.randint(0,100))
        reward.append(self.agent.compute_exploitability(evaluate_num))

        return np.mean(reward)

    

    
