# -*- coding: utf-8 -*-
''' Implement Leduc Hold'em Round class
'''

from rlcard.games.nolimitholdem.round import NolimitholdemRound

class NolimitLeducholdemRound(NolimitholdemRound):
    ''' Round can call other Classes' functions to keep the game running
    '''

    def __init__(self, num_players, init_raise_amount):
        ''' Initilize the round class

        Args:
            raise_amount (int): the raise amount for each raise
            allowed_raise_num (int): The number of allowed raise num
            num_players (int): The number of players
        '''
        super(NolimitLeducholdemRound, self).__init__(num_players, init_raise_amount)

    def get_nolimit_legal_actions(self, players):
        ''' Obtain the legal actions for the curent player

        Args:
            players (list): The players in the game

        Returns:
           (list):  A list of legal actions
        '''
        full_actions = ['call', 'fold', 'check']

        # If the current chips are less than that of the highest one in the round, we can not check
        if self.raised[self.game_pointer] < max(self.raised):
            full_actions.remove('check')

        # If the current player has put in the chips that are more than others, we can not call
        if self.raised[self.game_pointer] == max(self.raised):
            full_actions.remove('call')

        # If the current player has no more chips after call, we cannot raise
        diff = max(self.raised) - self.raised[self.game_pointer]
        if players[self.game_pointer].in_chips + diff >= players[self.game_pointer].remained_chips:
            return full_actions

        # Append available raise amount to the action list
        min_raise_amount = max(self.raised) - self.raised[self.game_pointer] + self.current_raise_amount
        if min_raise_amount <= 0:
            raise ValueError("Raise amount {} should not be smaller or equal to zero".format(min_raise_amount))
        # If the player cannot provide min raise amount, he has to all-in.
        if players[self.game_pointer].in_chips + min_raise_amount >= players[self.game_pointer].remained_chips:
            full_actions.append(players[self.game_pointer].remained_chips - players[self.game_pointer].in_chips)
        else:
            for available_raise_amount in range(min_raise_amount, players[self.game_pointer].remained_chips - players[self.game_pointer].in_chips + 1):
                full_actions.append(available_raise_amount)

        return full_actions