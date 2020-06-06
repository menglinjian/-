import numpy as np
import collections
import torch
import os
import pickle
import copy
from rlcard.utils.utils import *
import random
import multiprocessing
from multiprocessing import Pool, Process, Queue

class CFRAgent():
    ''' Implement CFR algorithm
    '''

    def __init__(self, env, isAbs=True, model_path='./cfr_plus_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.env = env
        self.model_path = model_path
        self.isAbs = isAbs

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy =collections.defaultdict(np.array)
        
        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)
        self.iteration = 0
        self.oppoCV = None

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        
        # Firstly, tranvers tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.player_num):
            #print(player_id)
            self.env.init_game()
            prob = 1
            self.traverse_tree(max(0, self.iteration - 10), prob, player_id)


    def traverse_tree(self, w, prob, player_id):
        
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.player_num)
        obs, legal_actions,_ = self.get_state(current_player)
        action_probs = self.regret_matching(obs)

        if current_player == player_id:
            for action in legal_actions:
                action_prob = action_probs[action]
                self.env.step(action)
                utility = self.traverse_tree(w, prob, player_id)
                self.env.step_back()

                state_utility += action_prob * utility
                action_utilities[action] = utility[player_id]

            for action in legal_actions:
                self.regrets[obs][action] = self.regrets[obs][action] + action_utilities[action] - state_utility[current_player]

        else:
            if obs not in self.average_policy:
                self.average_policy[obs] = np.zeros(self.env.action_num)

            for action in legal_actions:
                action_prob = action_probs[action]

                prob_temp = prob * action_prob
                self.env.step(action)
                utility = self.traverse_tree(w, prob_temp, player_id)
                self.env.step_back()
                state_utility += utility

                self.average_policy[obs][action] += prob * action_prob

        return state_utility

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def compute_exploitability_mlprocess(self, eval_num, process_num):
        from multiprocessing import Manager
        multiprocessing.freeze_support()
        u = []
        p = Pool(process_num)
        for i in range(0, process_num):
            u.append(p.apply_async(self.compute_exploitability, args=(int(eval_num/process_num),)))
        p.close()
        p.join()

        for i in range(0, len(u)):
            u[i] = u[i].get()

        return np.mean(u)

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.action_num)

        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.action_num)
        if positive_regret_sum > 0:
            for action in range(self.env.action_num):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.action_num):
                action_probs[action] = 1.0 / self.env.action_num

        return action_probs

    def set_evalenv(self, eval_env):
        self.eval_env = eval_env

    def get_evalenv(self):
        return self.eval_env

    def get_policy(self, state, legal_actions, isEval=False, isSubgame=False, oppoCV = None):
        obs = self.state_to_obs(state)
        if isEval and not isSubgame:
            return self.action_probs(obs.tostring(), legal_actions, self.average_policy)
        if isEval:
            '''
            if state['round'] >= 1:
                import agents.subgame_resolving as subgame
                Subgame = subgame.subgame(self.eval_env, self, state['current_player'], oppoCV)
                policy, self.oppoCV = Subgame.resolve() 
                return self.action_probs(obs.tostring(), legal_actions, policy)
            else:
            '''
            return self.action_probs(obs.tostring(), legal_actions, self.average_policy)
        else:
            return self.action_probs(obs.tostring(), legal_actions, self.policy)

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy:
            action_probs = np.array([1.0/self.env.action_num for _ in range(self.env.action_num)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state, env=None):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        '''
        probs = self.get_policy(state, state['legal_actions'], isEval=True, oppoCV=self.oppoCV)
        action = np.random.choice(len(probs), p=probs)
        return action

    def compute_exploitability(self, eval_num):
        self.cards = [Card('S', 'J'), Card('H', 'J'), Card('S', 'Q'), Card('H', 'Q'), Card('S', 'K'), Card('H', 'K')]
        self.env.init_game()
        self.set_evalenv(self.env)
        palyers_range=[]
        Range = []
        
        for i in range(0,len(self.cards)):
            Range.append([self.cards[i], 1/6])

        for player_id in range(self.env.player_num):
            palyers_range.append(Range)
        
        utility = []
        for i in range(0, eval_num):
            for player_id in range(self.env.player_num):
                self.t = 0
                self.env.init_game()
                self.oppoCV_temp = []
                self.oppoCV_temp2 = []
                self.oppoCV_o = None
                for i in range(0,len(self.cards)):
                    self.oppoCV_temp.append(None)
                    self.oppoCV_temp2.append(None)
                #print(palyers_range[0][0:3])
                palyers_range_ = copy.deepcopy(palyers_range)
                self.oppo_last_action = -1
                payoffs = self.traverse_exp(player_id, palyers_range_)
                #print("final",palyers_range[0][0:3])
                #print('player:{},payoff:{}'.format(player_id,payoffs[player_id]))
                #exit()
                utility.append(payoffs[player_id])
        
        return np.mean(utility)

    def traverse_exp(self, player_id, palyers_range):
        self.t=self.t+1
        if self.env.is_over():
            #print(self.env.get_payoffs())
            return self.env.get_payoffs()
        temp = 0
        current_player = self.env.get_player_id()
        _, legal_actions, state = self.get_state(current_player)
        #print('current_player:{}, legal_actions:{}'.format(current_player, legal_actions))
        #print("in",palyers_range[current_player][0:3])
        if current_player != player_id:
            action_probs = self.get_policy(state, legal_actions, isEval=True, isSubgame=True, oppoCV=self.oppoCV_o)
            self.oppoCV_o = self.oppoCV
            action = np.random.choice(len(action_probs), p=action_probs)
            self.oppo_last_action = action
            self.oppo_last_legal_actions = legal_actions
            self.oppo_last_state = state
            self.last_eval_env = copy.deepcopy(self.eval_env)
            self.env.step(action)

        else:
            #更新对手range
            self.temp_eval_env = self.get_evalenv()
            Sum = 0
            handcards = state['hand']
            publiccards = state['public_cards']
            for i,obj in enumerate(palyers_range[1-current_player]):
                oppocards = [obj[0].get_index()]
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    palyers_range[1-current_player][i][-1] = 0
                    temp = i
                else:
                    if self.oppo_last_action == -1:
                        action_prob = 1
                    else:
                        self.set_evalenv(self.last_eval_env)
                        oppo_state = self.oppo_last_state
                        oppo_state['hand'] = oppocards
                        self.eval_env.game.players[1 - current_player].hand = [obj[0]]
                        action_prob = self.get_policy(oppo_state, self.oppo_last_legal_actions, isEval=True, isSubgame=True, oppoCV=self.oppoCV_temp2[i])[self.oppo_last_action]
                        self.oppoCV_temp2[i] = self.oppoCV
                    palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]*action_prob
                    Sum = Sum + palyers_range[1-current_player][i][-1]

            self.set_evalenv(self.temp_eval_env)
            for i,obj in enumerate(palyers_range[1-current_player]):
                palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]/Sum

            action = self.LocalBR(player_id, state, legal_actions, palyers_range[1-current_player])#对两人情况
            self.env.step(action)
        
        #print("out",palyers_range[current_player][0:3],"---",current_player, palyers_range[current_player][temp])
        #print("times:{},palyer:{},action:{}".format(self.t,current_player,action))
        return self.traverse_exp(player_id, palyers_range)

    def LocalBR(self, player_id, state, legal_actions, oppo_range):
        values = np.zeros(self.env.action_num)
        handcards = state['hand']
        publiccards = state['public_cards']
        pot_myself = state['all_chips'][0]
        pot_oppo = state['all_chips'][1]
        self.temp_eval_env = copy.deepcopy(self.get_evalenv())

        wp = self.WpRollout(player_id, handcards, publiccards, oppo_range)
        asked = pot_oppo - pot_myself 
        #print('asked',asked)
        values[0] = wp*pot_myself-(1-wp)*asked
        for action in legal_actions:
            if action >=2 :
                fp = 0
                oppo_range_temp = copy.deepcopy(oppo_range)
                Sum = 0
                self.env.step(action)
                _, oppo_legal_actions,state_ = self.get_state(1-player_id)
                #print(action, oppo_legal_actions)
                diff = state_['all_chips'][1] - pot_myself
                oppo_state = state_
                #print(state_['all_chips'],pot_myself+action-2)
                for i,obj in enumerate(oppo_range_temp):
                    oppocards = [obj[0].get_index()]
                    prob = obj[-1]
                    if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                        oppo_range_temp[i][-1] = 0
                    else:
                        oppo_state['hand'] = oppocards
                        self.eval_env = copy.deepcopy(self.env)
                        self.eval_env.game.players[1 - player_id].hand = [obj[0]]
                        foldprob = self.get_policy(oppo_state, oppo_legal_actions, isEval=True, isSubgame=True, oppoCV=self.oppoCV_temp[i])[1]
                        self.oppoCV_temp[i] = self.oppoCV
                        fp = fp + prob*foldprob
                        oppo_range_temp[i][-1] = oppo_range_temp[i][-1]*(1-foldprob)
                        Sum = Sum + oppo_range_temp[i][-1]
                for i,obj in enumerate(oppo_range_temp):
                    oppo_range_temp[i][-1] = oppo_range_temp[i][-1]/Sum
                
                self.env.step_back()

                wp = self.WpRollout(player_id, handcards, publiccards, oppo_range_temp)
                values[action] = fp*pot_myself + (1-fp)*(wp*(pot_myself+diff)-(1-wp)*(asked+diff))
                #print("fp",fp," Sum",Sum,'value',values[action])
                #print("action:",action,'--',values[action])
        result = np.argmax(values)
        self.set_evalenv(self.temp_eval_env)
        if values[result]>0:
            return result
        else:
            return 1#flod

    def WpRollout(self, player_id, handcards, publiccards, oppo_range):
        from rlcard.games.limitholdem.utils import compare_hands
        wp = 0 
        handcards = handcards
        publiccards = publiccards
        for i,obj in enumerate(oppo_range):
            oppocards = [obj[0].get_index()]
            '''
            print('-------')
            print(handcards)
            print(oppocards)
            print(publiccards)
            print(len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)))
            print('-------')
            '''
            prob = obj[-1]
            if prob!=0:
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    pass
                else:
                    cards = self.cards
                    #print('enter')
                    if len(publiccards)<1:
                        cards = [i.get_index() for i in cards if (i.get_index() not in handcards and i.get_index() not in oppocards)]
                        for c in cards:
                            #print(c)
                            publiccards_temp = publiccards + [c]
                            selfcards_temp = handcards 
                            oppocards_temp = oppocards
                            wp = wp + 0.25*prob*self.compare_hands(selfcards_temp, oppocards_temp, publiccards_temp)[0]
                    else:
                        publiccards_temp = publiccards
                        selfcards_temp = handcards 
                        oppocards_temp = oppocards 
                        wp = wp + prob*self.compare_hands(selfcards_temp, oppocards_temp, publiccards_temp)[0]
        return wp

    def compare_hands(self, selfcards, oppocards, publiccards):
        from rlcard.utils.utils import rank2int
        # Judge who are the winners
        winners = [0, 0]
        if sum(winners) < 1:
            if selfcards[0][1] == oppocards[0][1]:
                winners = [1, 1]
        if sum(winners) < 1:
            if selfcards[0][1] == publiccards[0][1]:
                winners[0] = 1
            elif oppocards[0][1] == publiccards[0][1]:
                winners[1] = 1
        if sum(winners) < 1:
            winners = [1, 0] if rank2int(selfcards[0][1]) > rank2int(oppocards[0][1]) else [0, 1]
        return winners

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        obs = self.state_to_obs(state)
        return obs.tostring(), state['legal_actions'],state

    def state_to_obs(self, state):
        public_cards = state['public_cards']
        hand = state['hand']
        idx = [self.env.card2index[card] for card in public_cards]
        obs = np.zeros(8)
        obs[idx] = 1
        idx = [self.env.card2index[card] for card in hand]
        obs[idx] = 2
        if self.isAbs == True:
            if state['all_chips'][0]>state['all_chips'][1]:
                obs[6] = 1
            else:
                obs[7] = 1
        else:
            obs[6] = state['all_chips'][0]
            obs[7] = state['all_chips'][1]
        return obs

    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()

