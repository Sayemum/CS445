""" Unit tests for the Card class. (A warm-up class for 
Object-Oriented Python.)

Author: Nathan Sprague
Version: 1/18/2021

"""

import unittest

from card import Card


class TestCard(unittest.TestCase):

    def test_ranks_class_variable(self):
        self.assertEqual(Card.RANKS, [None, "Ace", "2", "3", "4", "5",
                                      "6", "7", "8", "9", "10", "Jack",
                                      "Queen", "King"])
        card = Card(1, 1)
        self.assertEqual(card.RANKS, [None, "Ace", "2", "3", "4", "5",
                                      "6", "7", "8", "9", "10", "Jack",
                                      "Queen", "King"])

    def test_suits_class_variable(self):
        self.assertEqual(Card.SUITS, ["Clubs", "Diamonds", "Hearts", "Spades"])
        card = Card(1, 1)
        self.assertEqual(card.SUITS, ["Clubs", "Diamonds", "Hearts", "Spades"])

    def test_rank_and_suit(self):
        card = Card(2, 3)
        self.assertEqual(card.rank, 2)
        self.assertEqual(card.suit, 3)

        with self.assertRaises(AttributeError):
            card.rank = 7

        with self.assertRaises(AttributeError):
            card.suit = 7

    def test_position(self):
        two_hearts = Card(2, 2)
        ace_spades = Card(1, 3)
        three_diamonds = Card(3, 1)

        self.assertEqual(two_hearts.position, 27)
        self.assertEqual(ace_spades.position, 39)
        self.assertEqual(three_diamonds.position, 15)

        with self.assertRaises(AttributeError):
            two_hearts.position = 7

    def test_equals(self):
        ace_diamonds1 = Card(1, 1)
        ace_diamonds2 = Card(1, 1)
        ace_hearts = Card(1, 2)
        two_diamonds = Card(2, 1)
        self.assertTrue(ace_diamonds1 == ace_diamonds2)
        self.assertFalse(ace_diamonds1 != ace_diamonds2)
        self.assertFalse(ace_diamonds1 == ace_hearts)
        self.assertTrue(ace_diamonds1 != ace_hearts)
        self.assertFalse(ace_diamonds1 == two_diamonds)
        self.assertTrue(ace_diamonds1 != two_diamonds)

    def test_comparisons(self):
        two_hearts1 = Card(2, 2)
        two_hearts2 = Card(2, 2)
        ace_spades = Card(1, 3)
        three_diamonds = Card(3, 1)

        self.assertFalse(two_hearts1 < two_hearts2)
        self.assertFalse(two_hearts1 > two_hearts2)
        self.assertTrue(two_hearts1 <= two_hearts2)
        self.assertTrue(two_hearts1 >= two_hearts2)

        self.assertFalse(ace_spades < two_hearts2)
        self.assertTrue(ace_spades > two_hearts2)
        self.assertFalse(ace_spades <= two_hearts2)
        self.assertTrue(ace_spades >= two_hearts2)

        self.assertTrue(three_diamonds < two_hearts2)
        self.assertFalse(three_diamonds > two_hearts2)
        self.assertTrue(three_diamonds <= two_hearts2)
        self.assertFalse(three_diamonds >= two_hearts2)

    def test_string(self):
        two_hearts = Card(2, 2)
        ace_spades = Card(1, 3)
        three_diamonds = Card(3, 1)

        self.assertEqual(str(two_hearts), "2 of Hearts")
        self.assertEqual(str(ace_spades), "Ace of Spades")
        self.assertEqual(str(three_diamonds), "3 of Diamonds")


if __name__ == '__main__':
    unittest.main()
