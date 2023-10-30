# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:37:22 2023

@author: evanu
"""
import numpy as np

def wd(array1, array2):
    # Calculate the cumulative distributions
    cdf1 = np.cumsum(array1)
    cdf2 = np.cumsum(array2)

    # Calculate the WD
    wd = np.sum(np.abs(cdf1 - cdf2))

    return wd

def rotate_matrix(matrix):
    return np.rot90(matrix, k=1)  # Rotate 90 degrees counterclockwise

def qmwd(P, Q):
    m, n = P.shape[0], P.shape[1]
    
    # Normalize the arrays to sum up to 1
    P = P.astype(np.float32) / np.sum(P)
    Q = Q.astype(np.float32) / np.sum(Q)
    
    c=5 # precision
    j=float("1e{}".format(c))
    P=(P*j).round().astype(int)
    Q=(Q*j).round().astype(int)
    

    # Calculate the Manhattan Wasserstein Distance between vectorized matrices
    WD1 = wd(P.flatten(), Q.flatten())

    # Calculate the Manhattan Wasserstein Distance between rotated matrices
    RP, RQ = rotate_matrix(P), rotate_matrix(Q)
    WD2 = wd(RP.flatten(), RQ.flatten())

    # Calculate the Manhattan Wasserstein Distance between transposed matrices
    PT, QT = P.T, Q.T
    WD3 = wd(PT.flatten(), QT.flatten())

    # Calculate QMWD
    QMWD = max((WD1 / n) + (WD1 % n), (WD2 / m) + (WD2 % m), (WD3 / m) + (WD3 % m))

    return QMWD/j

# Example usage:
m = 100
n = 100

P = np.random.randint(0, 10, size=(m, n))
Q = np.random.randint(0, 10, size=(m, n))

print("Quasi Manhattan Wasserstein Distance:", qmwd(P, Q))
