# Write the distance_loop method
#import math

def distance_loop(x1, x2):
    """ Returns the Euclidean distance between the 1-d numpy arrays x1 and x2"""
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    # return math.sqrt(math.pow((x1[0]-x2[0]), 2) + math.pow((x1[1]-x2[1]), 2))
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i]-x2[i]) ** 2

    distance = distance ** (1/2)
    
    return distance

import numpy as np
import math

a = np.array([1.0, 2.0, 3.0])
b = np.array([0.0, 1.0, 4.0])

test_distance = distance_loop(a, b) 
expected_answer = 1.7320508075688772