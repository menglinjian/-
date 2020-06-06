from rlcard.core import Card
from rlcard.games.nolimitholdem.dealer import NolimitholdemDealer

class NolimitLeducholdemDealer(NolimitholdemDealer):

    def __init__(self):
        ''' Initialize a leducholdem dealer class
        '''
        self.deck = [Card('S', 'J'), Card('H', 'J'), Card('S', 'Q'), Card('H', 'Q'), Card('S', 'K'), Card('H', 'K')]
        self.shuffle()
        self.pot = 0
