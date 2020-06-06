import numpy as np
import collections
import torch
import os
import pickle
import copy
from rlcard.utils.utils import *
import random

class CFRAgent():
    ''' Implement CFR algorithm
    '''

    def __init__(self, env, model_path='./cfr_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.env = env
        self.model_path = model_path

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy =collections.defaultdict(np.array)
        
        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)
        self.iteration = 0

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        
        # Firstly, tranvers tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.player_num):
            #print(player_id)
            self.env.init_game()
            probs = np.ones(self.env.player_num)
            self.traverse_tree(probs, player_id)

        # Update policy
        self.update_policy()

    def traverse_tree(self, probs, player_id):
        ''' Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value

        Returns:
            state_utilities (list): The expected utilities for all the players
        '''
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.player_num)
        obs, legal_actions,_ = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)
        print(legal_actions)
        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # If it is current player, we record the policy and compute regret
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.action_num)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.action_num)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (action_utilities[action][current_player]
                    - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob
        return state_utility

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
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

    def get_policy(self, state, legal_actions, isEval=False):
        public_card = state['public_cards']
        hand = state['hand']
        cards = [] + hand
        if public_card:
            cards.append(public_card[0])
        idx = [self.env.card2index[card] for card in cards]
        obs = np.zeros(54)
        obs[idx] = 1
        if isEval:
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

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        '''
        probs = self.action_probs(state['obs'].tostring(), state['legal_actions'], self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        return action

    def compute_exploitability(self, eval_num):
        self.cards = [c.get_index() for c in init_standard_deck()]
        self.env.init_game()
        palyers_range=[]
        Range = []
        
        for i in range(0,52):
            for j in range(i+1,52):
                Range.append([self.cards[i], self.cards[j],1/1326])

        for player_id in range(self.env.player_num):
            palyers_range.append(Range)
        
        utility = []
        for i in range(0, eval_num):
            for player_id in range(self.env.player_num):
                self.t = 0
                self.env.init_game()
                #print(palyers_range[0][0:3])
                palyers_range_ = copy.deepcopy(palyers_range)
                self.oppo_last_action = -1
                payoffs = self.traverse_exp(player_id, palyers_range_)
                #print("final",palyers_range[0][0:3])
                #print('player:{},payoff:{}'.format(player_id,payoffs[player_id]*2))
                #exit()
                utility.append(payoffs[player_id]*2)
        
        return np.mean(utility)

    def traverse_exp(self, player_id, palyers_range):
        self.t=self.t+1
        if self.env.is_over():
            #print(self.env.get_payoffs())
            return self.env.get_payoffs()
        temp = 0
        current_player = self.env.get_player_id()
        _, legal_actions, state = self.get_state(current_player)

        #print("in",palyers_range[current_player][0:3])
        if current_player != player_id:
            action_probs = self.get_policy(state, legal_actions, isEval=True)
            action = np.random.choice(len(action_probs), p=action_probs)
            self.oppo_last_action = action
            self.oppo_last_legal_actions = legal_actions
            self.oppo_last_chips = state['all_chips']
            self.env.step(action)

        else:
            #更新对手range
            Sum = 0
            handcards = state['hand']
            publiccards = state['public_cards']
            for i,obj in enumerate(palyers_range[1-current_player]):
                oppocards = obj[0:2]
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    palyers_range[1-current_player][i][-1] = 0
                    temp = i
                else:
                    if self.oppo_last_action == -1:
                        action_prob = 1
                    else:
                        oppo_state = {}
                        oppo_state['hand'] = oppocards
                        oppo_state['public_cards'] = publiccards
                        oppo_state['all_chips'] = self.oppo_last_chips
                        action_prob = self.get_policy(oppo_state, self.oppo_last_legal_actions, isEval=True)[self.oppo_last_action]
                    palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]*action_prob
                    Sum = Sum + palyers_range[1-current_player][i][-1]

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

        wp = self.WpRollout(player_id, handcards, publiccards, oppo_range)
        asked = pot_oppo-pot_myself 
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
                self.env.step_back()
                oppo_state = {}
                diff = state_['all_chips'][1] - pot_myself
                oppo_state['public_cards'] = publiccards
                oppo_state['all_chips'] = state_['all_chips']
                #print(state_['all_chips'],pot_myself+action-2)
                for i,obj in enumerate(oppo_range_temp):
                    oppocards = obj[0:2]
                    prob = obj[-1]
                    if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                        oppo_range_temp[i][-1] = 0
                    else:
                        oppo_state['hand'] = oppocards
                        foldprob = self.get_policy(oppo_state, oppo_legal_actions, isEval=True)[1]
                        fp = fp + prob*foldprob
                        oppo_range_temp[i][-1] = oppo_range_temp[i][-1]*(1-foldprob)
                        Sum = Sum + oppo_range_temp[i][-1]
                for i,obj in enumerate(oppo_range_temp):
                    oppo_range_temp[i][-1] = oppo_range_temp[i][-1]/Sum

                wp = self.WpRollout(player_id, handcards, publiccards, oppo_range_temp)
                values[action] = fp*pot_myself + (1-fp)*(wp*(pot_myself+diff)-(1-wp)*(asked+diff))
                #print("fp",fp," Sum",Sum,'value',values[action])
                #print("action:",action,'--',values[action])
        result = np.argmax(values)
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
            oppocards = obj[0:2]
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
                    if len(publiccards)<5:
                        cards = [i for i in cards if (i not in publiccards and i not in handcards and i not in oppocards)]
                        for i in range(0,20):
                            publiccards_temp = publiccards + random.sample(cards,5-len(publiccards))
                            selfcards_temp = handcards + publiccards_temp
                            oppocards_temp = oppocards + publiccards_temp
                            wp = wp + 0.05*prob*compare_hands(selfcards_temp, oppocards_temp)[0]
                    else:
                        publiccards_temp = publiccards
                        selfcards_temp = handcards + publiccards_temp
                        oppocards_temp = oppocards + publiccards_temp 
                        wp = wp + prob*compare_hands(selfcards_temp, oppocards_temp)[0]
        return wp


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
        #print(state)
        return state['obs'].tostring(), state['legal_actions'],state

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

