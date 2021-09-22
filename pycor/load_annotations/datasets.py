from pycor.utils.preprocess import clean_ddo_bet


class DataInstance:
    def __init__(self, lemma, ddo_bet, cor, vector,
                 onto=None, frame=None, score=0, figurative=0):
        self.lemma = lemma
        self.ddo_bet = ddo_bet
        self.cor = cor
        self.vector = vector
        self.onto = onto
        self.frame = frame
        self.score = score
        self.figurative = figurative
        self.main_sense = clean_ddo_bet(self.ddo_bet)
