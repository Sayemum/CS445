class Card:
    
    RANKS = [None, "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
    
    SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
    
    """Constructs a card of the given rank and suit."""
    def __init__(self, rank, suit):
        self.__rank = rank
        self.__suit = suit
    
    def __lt__(self, that):
        if self.suit < that.suit:
            return True
        if self.suit > that.suit:
            return False
        return self.rank < that.rank
    
    def __le__(self, that):
        return self < that or self == that
    
    def __eq__(self, that):
        return self.rank == that.rank and self.suit == that.suit
    
    def __ne__(self, that):
        return self.rank != that.rank or self.suit != that.suit
    
    def __gt__(self, that):
        if self.suit > that.suit:
            return True
        if self.suit < that.suit:
            return False
        return self.rank > that.rank
    
    def __ge__(self, that):
        return self > that or self == that
    
    
    """Gets the card's rank."""
    @property
    def rank(self):
        return self.__rank
    
    
    """Gets the card's suit."""
    @property
    def suit(self):
        return self.__suit
    
    
    """Returns the card's index in a sorted deck of 52 cards."""
    @property
    def position(self):
        return self.suit * 13 + self.rank - 1
    
    
    """Returns a string representation of the card."""
    def __str__(self):
        return self.RANKS[self.rank] + " of " + self.SUITS[self.suit]