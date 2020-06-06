from rlcard.utils.utils import rank2int

class NolimitLeducholdemJudger(object):
    ''' The Judger class for Leduc Hold'em
    '''

    def __init__(self):
        ''' Initialize a judger class
        '''
        super().__init__()

    @staticmethod
    def judge_game(players, public_cards):
        ''' Judge the winner of the game.

        Args:
            players (list): The list of players who play the game
            public_card (object): The public card that seen by all the players

        Returns:
            (list): Each entry of the list corresponds to one entry of the
        '''
        # Judge who are the winners
        winners = [0, 0]
        public_card = public_cards[0] if public_cards!=[] else None
        # If one player folds, the other player is the winner
        for idx, player in enumerate(players):
                if player.status == 'folded':
                    winners[(idx+1)%2] = 1
                    break
        if sum(winners) < 1:
            if players[0].hand[0].rank == players[1].hand[0].rank:
                winners = [1, 1]
        if sum(winners) < 1:
            for idx, player in enumerate(players):
                if player.hand[0].rank == public_card.rank:
                    winners[idx] = 1
                    break
        if sum(winners) < 1:
            winners = [1, 0] if rank2int(players[0].hand[0].rank) > rank2int(players[1].hand[0].rank) else [0, 1]

        # Compute the total chips
        total = 0
        for p in players:
            total += p.in_chips

        each_win = float(total) / sum(winners)

        payoffs = []
        for i, _ in enumerate(players):
            if winners[i] == 1:
                payoffs.append(each_win - players[i].in_chips)
            else:
                payoffs.append(float(-players[i].in_chips))

        return payoffs
