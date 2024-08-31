from functools import total_ordering


@total_ordering
class Card:
    
    RANKS = [None, "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
    
    SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
    
    #rank = None
    #suit = None
    
    """Constructs a card of the given rank and suit."""
    def __init__(self, rank, suit):
        self.__rank = rank
        self.__suit = suit
    
    
    """
    * Returns a negative integer if this card comes before
     * the given card, zero if the two cards are equal, or
     * a positive integer if this card comes after the card.
    """
    # def compareTo(self, that):
    #     if self.suit < that.suit:
    #         return -1
    #     if self.suit > that.suit:
    #         return 1
    #     if self.rank < that.rank:
    #         return -1
    #     if self.rank < that.rank:
    #         return 1
    #     return 0
    
    def __lt__(self, that):
        return compare(self, that)
    
    def __le__(self, that):
        return compare(self, that)
    
    def __eq__(self, that):
        #return compare(self, that)
        return self.equals(that)
    
    def __ne__(self, that):
        return compare(self, that)
    
    def __gt__(self, that):
        return compare(self, that)
    
    def __ge__(self, that):
        return compare(self, that)
    
    
    """
    * Returns true if the given card has the same
     * rank AND same suit; otherwise returns false.
    """
    def equals(self, that):
        return self.getRank == that.getRank and self.getSuit == that.getSuit
    
    
    """Gets the card's rank."""
    @property
    def getRank(self):
        return self.__rank
    
    
    """Gets the card's suit."""
    @property
    def getSuit(self):
        return self.__suit
    
    
    """Returns the card's index in a sorted deck of 52 cards."""
    def position(self):
        return self.getSuit * 13 + self.getRank - 1
    
    
    """Returns a string representation of the card."""
    def __str__(self):
        return self.RANKS[self.getRank] + " of " + self.SUITS[self.getSuit]


# helper method for comparison
def compare(first, second):
    firstSuit = first.getSuit
    firstRank = first.getRank
    secondSuit = second.getSuit
    secondRank = second.getRank
    
    if firstSuit < secondSuit:
        return -1
    if firstSuit > secondSuit:
        return 1
    if firstRank < secondRank:
        return -1
    if firstRank < secondRank:
        return 1
    return 0