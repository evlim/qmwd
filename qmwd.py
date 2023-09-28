import numpy as np
import cvxpy as cp

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

    return QMWD


def mwd(P, Q):
    m, n = P.shape
    cost_matrix = np.zeros((m*n, m*n))
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    cost_matrix[i*n+j, k*n+l] = abs(i-k) + abs(j-l)
    gamma = cp.Variable((m*n, m*n))
    objective = cp.Minimize(cp.sum(cp.multiply(cost_matrix, gamma)))
    constraints = [gamma >= 0, cp.sum(gamma, axis=0) == P.ravel(), cp.sum(gamma, axis=1) == Q.ravel()]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return prob.value





# Example usage:
m = 2
n = 300

while True:
    P = np.random.randint(0, 10, size=(m, n))
    Q = np.random.randint(0, 10, size=(m, n))
    if P.sum() == Q.sum():
        break


print("Wasserstein Distance (Flatten):", wd(P.flatten(), Q.flatten()))
print("Quasi Manhattan Wasserstein Distance:", qmwd(P, Q))
print("Manhattan Wasserstein Distance: ", mwd(P, Q))