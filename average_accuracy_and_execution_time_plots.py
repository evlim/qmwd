import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import timeit

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

# Initialize lists to store accuracy values and execution times
avg_accuracies_wd = []
avg_accuracies_qmwd = []
avg_execution_times_wd = []
avg_execution_times_qmwd = []
avg_execution_times_mwd = []

num_repeats = 20  # Number of times to repeat the calculations

# Example usage:
m_values = list(range(2, 31))  # m from 2 to 30
n = 30

for m in m_values:
    accuracies_wd = []
    accuracies_qmwd = []
    execution_times_wd = []
    execution_times_qmwd = []
    execution_times_mwd = []

    for _ in range(num_repeats):
        while True:
            P = np.random.randint(0, 10, size=(m, n))
            Q = np.random.randint(0, 10, size=(m, n))
            if P.sum() == Q.sum():
                break

        print("m=", m)

        # Calculate accuracy and execution times
        start_time = timeit.default_timer()
        wd_result = wd(P.flatten(), Q.flatten())
        execution_time_wd = timeit.default_timer() - start_time
        accuracy_wd = abs(mwd(P, Q) - wd_result) / mwd(P, Q)

        start_time = timeit.default_timer()
        qmwd_result = qmwd(P, Q)
        execution_time_qmwd = timeit.default_timer() - start_time
        accuracy_qmwd = abs(mwd(P, Q) - qmwd_result) / mwd(P, Q)

        execution_time_mwd = timeit.timeit(lambda: mwd(P, Q), number=1)

        accuracies_wd.append(accuracy_wd)
        accuracies_qmwd.append(accuracy_qmwd)
        execution_times_wd.append(execution_time_wd)
        execution_times_qmwd.append(execution_time_qmwd)
        execution_times_mwd.append(execution_time_mwd)

    # Calculate the average values for this specific m
    avg_accuracy_wd = np.mean(accuracies_wd)
    avg_accuracy_qmwd = np.mean(accuracies_qmwd)
    avg_execution_time_wd = np.mean(execution_times_wd)
    avg_execution_time_qmwd = np.mean(execution_times_qmwd)
    avg_execution_time_mwd = np.mean(execution_times_mwd)

    avg_accuracies_wd.append(avg_accuracy_wd)
    avg_accuracies_qmwd.append(avg_accuracy_qmwd)
    avg_execution_times_wd.append(avg_execution_time_wd)
    avg_execution_times_qmwd.append(avg_execution_time_qmwd)
    avg_execution_times_mwd.append(avg_execution_time_mwd)

# Print and plot the average results
print("Average Accuracies (WD):", avg_accuracies_wd)
print("Average Accuracies (QMWD):", avg_accuracies_qmwd)
print("Average Execution Times (WD):", avg_execution_times_wd)
print("Average Execution Times (QMWD):", avg_execution_times_qmwd)
print("Average Execution Times (MWD):", avg_execution_times_mwd)

# Plot the average accuracies
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(m_values, avg_accuracies_qmwd, label='QMWD', linestyle='-')
plt.plot(m_values, avg_accuracies_wd, label='WD', linestyle='--')
plt.xlabel('m')
plt.ylabel('Error')
plt.legend()
plt.title('Error vs. m')

# Plot the average execution times
plt.subplot(1, 2, 2)
plt.plot(m_values, avg_execution_times_qmwd, label='QMWD', linestyle='-')
plt.plot(m_values, avg_execution_times_wd, label='WD', linestyle='--')
plt.plot(m_values, avg_execution_times_mwd, label='MWD', linestyle=':')
plt.xlabel('m')
plt.ylabel('Time (s)')
plt.legend()
plt.title('Execution Time vs. m')

plt.tight_layout()

# Save the plots as SVG files
plt.savefig('average_accuracy_and_execution_time_plots.svg', format='svg')

plt.show()
