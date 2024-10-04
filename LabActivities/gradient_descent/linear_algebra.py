"""CS445 Linear Algebra Exercises

Answer key using numpy.

Author: Sayemum Hassan
"""

import numpy as np


def main():
    B = np.array([[1,2,-3], [3,4,-1]])
    A = np.array([[2, -5, 1], [1, 4, 5], [2, -1, 6]])
    y = np.array([[2], [-4], [1]])
    z = np.array([[-15], [-8], [-22]])
    
    # Question 1
    result1 = B @ A
    print("\nRESULT 1:")
    print(result1)
    
    # Question 2
    result2 = A @ (B.transpose())
    print("\nRESULT 2:")
    print(result2)
    
    # Question 3
    result3 = A @ y
    print("\nRESULT 3:")
    print(result3)
    
    # Question 4
    result4 = (y.transpose()) @ z
    print("\nRESULT 4:")
    print(result4)
    
    # Question 5
    result5 = y @ (z.transpose())
    print("\nRESULT 5:")
    print(result5)
    
    # Question 6
    A = np.array([[1, 2], [3, 0]])
    b = np.array([[4], [6]])
    
    inversedA = np.linalg.inv(A)
    checkA = A @ inversedA
    
    print("\nINVERSED A:")
    print(inversedA)
    print(checkA)
    
    # Question 7
    x = inversedA @ b
    Ax = A @ x
    print("\nx:")
    print(x)
    print("\nAx:")
    print(Ax)


if __name__ == "__main__":
    main()
