import json
import os
import numpy as np

import rlcard
from rlcard.envs.env import Env
from game import LimitholdemGame as Game
from rlcard.utils.utils import *
from rlcard import models


class LimitholdemEnv(Env):
    ''' Leduc Hold'em Environment
    '''

    def __init__(self, allow_step_back=False,chips=100):
        ''' Initialize the Limitholdem environment
        '''
        super().__init__(Game(allow_step_back,chips), allow_step_back)
        self.actions = ['call', 'fold', 'check','rasie']
        self.state_shape = [54]
        for raise_amount in range(1, self.game.init_chips+1):
            self.actions.append(raise_amount)
        #print(self.actions)  
        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)
        self.action_num = 5

    def get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        processed_state = {}
        #print(state['legal_actions']) (str)
        legal_actions = [self.actions.index(a) for a in state['legal_actions']]
        processed_state['legal_actions'] = legal_actions
        action_history = np.zeros((2,8))
        for i in range(2):
            for j in range(0,len(self.game.action_history[i])):
                action_history[i][j] = self.actions.index(self.game.action_history[i][j])+1
        

        l=[]
        for item in legal_actions:
            if item>4:
                break
            else:
                l.append(item)
        
        processed_state['legal_actions'] = l#egal_actions
        
        public_cards = state['public_cards']
        hand = state['hand']
        #cards = public_cards + hand
        idx = [self.card2index[card] for card in public_cards]
        obs = np.zeros(54)
        obs[idx] = 1
        idx = [self.card2index[card] for card in hand]
        obs[idx] = 2
        obs[52] = state['all_chips'][0]
        obs[53] = state['all_chips'][1]
        processed_state['obs'] = obs
        processed_state['my_chips'] = state['my_chips']
        processed_state['public_cards'] = state['public_cards']
        processed_state['all_chips'] = state['all_chips']
        processed_state['hand'] = state['hand']
        processed_state['current_player'] = state['player_id']
        processed_state['round'] = self.game.round_counter
        processed_state['action_history'] = action_history
        #print("current player:{},hands:{},publicards:{}".format(state['player_id'],state['hand'],state['public_cards']))
        return processed_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def run(self, is_training=False, seed=None):
        ''' Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.
            seed (int): A seed for running the game. For single-process program,
              the seed should be set to None. For multi-process program, the
              seed should be asigned for reproducibility.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        if self.single_agent_mode or self.human_mode:
            raise ValueError('Run in single agent mode or human mode is not allowed.')

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.init_game()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs

    def print_state(self, player,legal_actions):
        ''' Print out the state of a given player

        Args:
            player (int): Player id
        '''
        state = self.game.get_state(player)
        print('\n=============== Community Card ===============')
        print_card(state['public_cards'])
        print('===============   Your Hand    ===============')
        print_card(state['hand'])
        print('===============     Chips      ===============')
        print('Yours:   ', end='')
        for _ in range(state['my_chips']):
            print('+', end='')
        print('')
        for i in range(self.player_num):
            if i != self.active_player:
                print('Agent {}: '.format(i) , end='')
                for _ in range(state['all_chips'][i]):
                    print('+', end='')
        print('\n=========== Actions You Can Choose ===========')
        print(', '.join([self.actions[action] + ': ' + str(action) for action in legal_actions]))
        print('')
