"""This program generates random text based on n-grams
calculated from sample text.

Author: Nathan Sprague and Sayemum Hassan
Date: 8/21/24
Modified: 8/31/24

"""

# Honor code statement (if you received help from an outside source):
#

import random
import string
from typing import Dict, List, Tuple
from collections import defaultdict

# Create some type aliases to simplify the type hinting below.
BigramDict = Dict[str, Dict[str, float]]
TrigramDict = Dict[Tuple[str, str], Dict[str, float]]


def text_to_list(file_name: str) -> List[str]:
    """Convert the provided plain-text file to a list of words.  All
    punctuation will be removed, and all words will be converted to
    lower-case.

    Args:
        file_name: A string containing a file path.

    Returns:
        A list containing the words from the file.
    """
    with open(file_name, "r") as handle:
        text = handle.read().lower()
        text = text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
    return text.split()


def calculate_unigrams(word_list: List[str]) -> Dict[str, float]:
    """Calculate the probability distribution over individual words.

    Args:
       word_list: a list of strings corresponding to the sequence of
           words in a document. Words must be all lower-case with no
           punctuation.

    Returns:
       A dictionary mapping from words to probabilities.

    Example:

    >>> calculate_unigrams(['i', 'think', 'therefore', 'i', 'am'])
    {'i': 0.4, 'am': 0.2, 'think': 0.2, 'therefore': 0.2}

    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    word_counts = {}
    
    for word in word_list:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    total_words = len(word_list)
    
    prob_distributions = {word: (count / total_words) for word, count in word_counts.items()}
    
    return prob_distributions


def calculate_bigrams(word_list: List[str]) -> BigramDict:
    """Calculate, for each word in the list, the probability distribution
    over possible subsequent words.

    This function returns a dictionary that maps from words to
    dictionaries that represent probability distributions over
    subsequent words.

    Args:
       word_list: a list of strings corresponding to the sequence of
           words in a document. Words must be all lower-case with no
           punctuation.

    Example:

    >>> b = calculate_bigrams(['i', 'think', 'therefore', 'i', 'am',\
                               'i', 'think', 'i', 'think'])
    >>> print(b)
    {'i':  {'am': 0.25, 'think': 0.75},
     None: {'i': 1.0},
     'am': {'i': 1.0},
     'think': {'i': 0.5, 'therefore': 0.5},
     'therefore': {'i': 1.0}}

    Note that None stands in as the predecessor of the first word in
    the sequence.

    Once the bigram dictionary has been obtained it can be used to
    obtain distributions over subsequent words.

    >>> print(b['i'])
    {'am': 0.25, 'think': 0.75}

    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    bigram_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    
    prev_word = None
    
    for word in word_list:
        bigram_counts[prev_word][word] += 1
        total_counts[prev_word] += 1
        prev_word = word
    
    # convert counts to probabilities
    bigram_probabilities = {}
    for prev_word, next_words in bigram_counts.items():
        bigram_probabilities[prev_word] = {}
        total = total_counts[prev_word]
        
        for next_word, count in next_words.items():
            bigram_probabilities[prev_word][next_word] = count / total
    
    return bigram_probabilities


def calculate_trigrams(word_list: List[str]) -> TrigramDict:
    """Calculate, for each adjacent pair of words in the list, the
    probability distribution over possible subsequent words.

    The returned dictionary maps from two-word tuples to dictionaries
    that represent probability distributions over subsequent
    words.

    Example:

    >>> calculate_trigrams(['i', 'think', 'therefore', 'i', 'am',\
                                'i', 'think', 'i', 'think'])
    {('think', 'i'): {'think': 1.0},
    ('i', 'am'): {'i': 1.0},
    (None, None): {'i': 1.0},
    ('therefore', 'i'): {'am': 1.0},
    ('think', 'therefore'): {'i': 1.0},
    ('i', 'think'): {'i': 0.5, 'therefore': 0.5},
    (None, 'i'): {'think': 1.0},
    ('am', 'i'): {'think': 1.0}}
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    trigrams = {}
    bigram_counts = {}
    
    # insert None at the beginning
    word_list = [None, None] + word_list
    
    # count occurrences
    for i in range(len(word_list) - 2):
        bigram = (word_list[i], word_list[i+1])
        next_word = word_list[i+2]
        
        # update bigram counts
        if bigram not in bigram_counts:
            bigram_counts[bigram] = 0
        bigram_counts[bigram] += 1
        
        # update trigram counts
        if bigram not in trigrams:
            trigrams[bigram] = {}
        if next_word not in trigrams[bigram]:
            trigrams[bigram][next_word] = 0
        trigrams[bigram][next_word] += 1
        
    # convert counts to probabilities
    for bigram, next_word_counts in trigrams.items():
        total_count = bigram_counts[bigram]
        trigrams[bigram] = {word: count / total_count for word, count in next_word_counts.items()}
    
    return trigrams


def random_unigram_text(unigrams: Dict[str, float], num_words: int) -> str:
    """Generate a random sequence according to the provided probabilities.

    Args:
       unigrams: Probability distribution over words (as returned by
           the calculate_unigrams function).
       num_words: The number of words of random text to generate.

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    Example:

    >>> u = calculate_unigrams(['i', 'think', 'therefore', 'i', 'am'])
    >>> random_unigram_text(u, 5)
    'think i therefore i i'

    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    words = list(unigrams.keys())
    probabilities = list(unigrams.values())
    
    random_words = random.choices(words, probabilities, k=num_words)
    
    random_text = ' '.join(random_words)
    
    return random_text


def random_bigram_text(first_word: str, bigrams: BigramDict, num_words: int) -> str:
    """Generate a random sequence of words following the word pair
    probabilities in the provided distribution.

    Args:
       first_word: This word will be the first word in the generated
           text.
       bigrams: Probability distribution over word pairs (as returned
           by the calculate_bigrams function).
       num_words: Length of the generated text (including the provided
          first word)

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    Example:
    >>> b = calculate_bigrams(['i', 'think', 'therefore', 'i', 'am',\
                               'i', 'think', 'i', 'think'])
    >>> random_bigram_text('think', b, 5)
    'think i think therefore i'

    >>> random_bigram_text('think', b, 5)
    'think therefore i think therefore'

    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    current_word = first_word
    text = [current_word]
    
    for _ in range(num_words - 1):
        next_words = bigrams.get(current_word, None)
        if not next_words:
            break
        words = list(next_words.keys())
        probabilities = list(next_words.values())
        current_word = random.choices(words, probabilities)[0]
        text.append(current_word)
    
    return ' '.join(text)


def random_trigram_text(
    first_word: str, second_word: str, trigrams: TrigramDict, num_words: int
) -> str:
    """Generate a random sequence of words according to the provided
    trigram distributions. The first two words provided must
    appear in the trigram distribution.

    Args:
       first_word: The first word in the generated text.
       second_word: The second word in the generated text.
       trigrams: trigram probabilities (as returned by the
           calculate_trigrams function).
       num_words: Length of the generated text (including the provided
           words)

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    # ensure we have enough words
    if num_words < 2:
        raise ValueError("num_words must be at least 2")
    
    # check if initial bigram exists in trigrams
    if (first_word, second_word) not in trigrams:
        raise ValueError("initial word pair doesn't exist in trigram model")
    
    # initialize result with first two words
    result = [first_word, second_word]
    
    for _ in range(num_words - 2):
        bigram = (result[-2], result[-1])
        
        if bigram not in trigrams:
            break
        
        # get next word probabilities
        next_words = trigrams[bigram]
        
        if not next_words:
            break
        
        # choose next words based on prob distribution
        next_word = random.choices(
            list(next_words.keys()),
            weights=next_words.values(),
            k=1
        )[0]
        
        result.append(next_word)
        
    return ' '.join(result)


def unigram_main():
    """Generate text from Huck Fin unigrams."""
    words = text_to_list("huck.txt")
    unigrams = calculate_unigrams(words)
    print(random_unigram_text(unigrams, 100))


def bigram_main():
    """Generate text from Huck Fin bigrams."""
    words = text_to_list("huck.txt")
    bigrams = calculate_bigrams(words)
    print(random_bigram_text("the", bigrams, 100))


def trigram_main():
    """Generate text from Huck Fin trigrams."""
    words = text_to_list("huck.txt")
    trigrams = calculate_trigrams(words)
    print(random_trigram_text("there", "is", trigrams, 100))


if __name__ == "__main__":
    # You can insert testing code here, or switch out the main method
    # to try bigrams or trigrams.
    # unigram_main()
    # bigram_main()
    trigram_main()
