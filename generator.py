from typing import List

def all_distances(numbers: List[float]) -> List[List[float]]:
    """Return a nxn two-dimensional list containing the pairwise distances
    between all n provided numbers, where entry (i, j) contains the absolute 
    value of the difference between the i'th and j'th entry in the list.

    >>> all_distances([0.0, 1.0, 3.0])
    [[0.0, 1.0, 3.0], 
     [1.0, 0.0, 2.0], 
     [3.0, 2.0, 0.0]]
    
    """
    # YOUR CODE HERE
    #raise NotImplementedError()
    n = len(numbers)

    # matrix = [[x] for x in range(n)]
    matrix = [0] * n
    #for i in range(n):
        #new_list = [n]
        #matrix.append()
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = abs(matrix[i] - matrix[j])

    return matrix

print(all_distances([0.0, 1.0, 3.0]))