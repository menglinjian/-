import numpy as np
from copy import copy
from copy import deepcopy
from dealer import NolimitLeducholdemDealer as Dealer
from player import NolimitLeducholdemPlayer as Player
from judger import NolimitLeducholdemJudger as Judger
from round import NolimitLeducholdemRound as Round

from rlcard.games.nolimitholdem.game import NolimitholdemGame

class NolimitLeducholdemGame(NolimitholdemGame):

    def __init__(self, allow_step_back=False,chips=100,samll_blind = 1):
        ''' Initialize the class nolimitholdem Game
        '''
        self.allow_step_back = allow_step_back

        # small blind and big blind
        self.small_blind = samll_blind
        self.big_blind = 2 * self.small_blind

        # config players
        self.num_players = 2
        self.init_chips = chips


    def init_game(self):
        ''' Initialilze the game of Limit Texas Hold'em

        This version supports two-player limit texas hold'em

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        self.dealer = Dealer()

        # Initilize two players to play the game
        self.players = [Player(i, self.init_chips) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger()

        # Deal cards to each  player to prepare for the first round
        for i in range(self.num_players):
            self.players[i].hand.append(self.dealer.deal_card())

        # Initilize public cards
        self.public_cards = []

        # Randomly choose a big blind and a small blind
        s = np.random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.big_blind
        self.players[s].in_chips = self.small_blind

        # The player next to the small blind plays the first
        self.game_pointer = (b + 1) % self.num_players

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(self.num_players, self.big_blind)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []
        self.action_history = []
        for i in range(2):
            self.action_history.append([])
        state = self.get_state(self.game_pointer)

        return state, self.game_pointer
    
    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''
        if self.allow_step_back:
            # First snapshot the current state
            r = deepcopy(self.round)
            b = self.game_pointer
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            ah = deepcopy(self.action_history)
            self.history.append((r, b, r_c, d, p, ps, ah))
        #self.action_history.append([action, self.game_pointer]) action(str)
        # Then we proceed to the next round

        if len(self.action_history[self.game_pointer])<8:
            self.action_history[self.game_pointer].append(action)
        #print(self.round_counter, self.game_pointer, '\n', self.action_history)

        self.game_pointer = self.round.proceed_round(self.players, action)
        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the first round, we deal 1 cards
            if self.round_counter == 0:
                self.public_cards.append(self.dealer.deal_card())

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)
        
        return state, self.game_pointer

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, self.players, self.action_history = self.history.pop()
            #print('back',self.round_counter, self.game_pointer, '\n', self.action_history)
            return True
        return False


    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        alive_players = [1 if p.status=='alive' else 0 for p in self.players]
        # If only one player is alive, the game is over.
        if sum(alive_players) == 1:
            return True

        # If all rounds are finshed
        if self.round_counter >= 2:
            return True
        return False

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        chips_payoffs = self.judger.judge_game(self.players, self.public_cards)
        payoffs = np.array(chips_payoffs)
        return payoffs

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_nolimit_legal_actions(self.players)

    def get_state(self, player):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()
        state = self.players[player].get_state(self.public_cards, chips, legal_actions)

        return state