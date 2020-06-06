''' An example of solve Leduc Hold'em with CFR
'''
import numpy as np
from rlcard.agents.random_agent import RandomAgent
import rlcard
import cfr_agent 
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from rlcard.envs.nolimitholdem import NolimitholdemEnv
# Make environment and enable human mode
chips = 8#20
raise_interval = 2
env = NolimitholdemEnv(allow_step_back=True,chips=chips)
eval_env = NolimitholdemEnv(chips=chips)

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 1#00
save_plot_every = 5#00
evaluate_num = 5#0
episode_num = 1#000

# The paths for saving the logs and learning curves
root_path = './experiments/nolimit_holdem_cfr_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'
log_reward_path = root_path + '_reward_log.txt'
csv_reward_path = root_path + '_reward_performance.csv'
# Set a global seed
set_global_seed(10)

# Initilize CFR Agent
agent = cfr_agent.CFRAgent(env)
#agent.load()  # If we have saved model, we first load the model

# Evaluate CFR against pre-trained NFSP
#eval_env.set_agents([agent, models.load('leduc-holdem-nfsp').agents[0]])
eval_env.set_agents([agent, RandomAgent(action_num=env.action_num)])
# Init a Logger to plot the learning curve
logger = Logger(xlabel='iteration', ylabel='exploitability', legend='CFR on nolimit Holdem', log_path=log_path, csv_path=csv_path)
logger_reward = Logger(xlabel='iteration', ylabel='reward', legend='CFR on nolimit Holdem', log_path=log_reward_path, csv_path=csv_reward_path)
for episode in range(episode_num):
    agent.train()
    if episode%1000 == 0:
        print('\rIteration {}'.format(episode), end='\n')
    # Evaluate the performance. Play with NFSP agents.
    if episode % evaluate_every == 0:
        #agent.save() # Save model
        reward = 0
        for eval_episode in range(evaluate_num):
            his, payoffs = eval_env.run(is_training=False)
            reward += payoffs[0]

        logger_reward.log('\n########## Evaluation ##########')
        logger_reward.log('Iteration: {} Average reward is {}'.format(episode, float(reward)/evaluate_num))

        # Add point to logger
        logger_reward.add_point(x=episode, y=float(reward)/evaluate_num)
        import time 
        start = time.perf_counter()
        exploitability = agent.compute_exploitability(evaluate_num)
        end = time.perf_counter()
        logger.log('episode: {} cost {:10}s ,exploitability is {}'.format(episode, end-start, exploitability))
        logger.add_point(x=episode, y=exploitability)
        print("\n")
    # Make plot
    if episode % save_plot_every == 0 and episode > 0:
        logger.make_plot(save_path=figure_path+str(episode)+'.png')
        logger_reward.make_plot(save_path=figure_path+str(episode)+'reward'+'.png')

# Make the final plot
agent.save()
logger.make_plot(save_path=figure_path+'final_'+str(episode)+'.png')
logger_reward.make_plot(save_path=figure_path+'final_'+str(episode)+'reward'+'.png')
