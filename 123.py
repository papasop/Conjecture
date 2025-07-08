import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi

# Parameters
x_range = np.linspace(10, 10000, 500)  # Extended x range
N_values = [50, 100, 150]  # Test different N values
C_values = [1000, 5000, 10000]  # Test different C values
epsilon_values = [1e-5, 1e-6, 1e-7]  # Test different epsilon values

# Define the modal collapse function
def phi(x, N):
    A_n = np.array([1 / n**2 for n in range(1, N+1)])  # Amplitude decay
    lambda_n = np.array([n * 0.01 for n in range(1, N+1)])  # Frequency
    theta_n = np.random.uniform(0, 2 * np.pi, N)  # Random phase
    phi_x = np.sum(A_n * np.cos(lambda_n * np.log(x) + theta_n))
    return phi_x

# Define the residual field
def delta(x, N=50):
    pi_x = primepi(int(x)) / x  # Prime counting function divided by x
    rho_x = 1 / np.log(x) + phi(x, N)  # Structural field
    delta_x = pi_x - rho_x  # Residual field
    return float(delta_x)  # Convert to Python float to avoid SymPy Float

# Define the entropy proxy
def entropy(x, N=50):
    delta_x = delta(x, N)
    delta_x = np.array(delta_x, dtype=np.float64)  # Explicitly set dtype
    return np.log(1 + delta_x**2)

# Conjecture 1: Structure Residual Boundedness
def conjecture_1(x_range, N=50, C=1000):
    residual_values = [delta(x, N) for x in x_range]
    residual_mean = np.mean(residual_values)
    residual_max = np.max(np.abs(residual_values))
    if residual_max < C / np.log(x_range[-1]):
        return 'Passed'
    else:
        return 'Failed'

# Conjecture 2: Collapse Entropy Attractor Principle
def conjecture_2(x_range, N=50, epsilon=1e-5):
    entropy_values = [entropy(x, N) for x in x_range]
    min_entropy = min(entropy_values)
    min_x = x_range[entropy_values.index(min_entropy)]
    if np.abs(min_x - x_range[0]) < epsilon:
        return min_entropy, min_x
    else:
        return min_entropy, None

# Conjecture 3: Spectral Mode Convergence Hierarchy
def conjecture_3(x_range, N=50):
    lambda_n = np.array([n * 0.01 for n in range(1, N+1)])  # Frequencies
    delta_lambda_values = np.diff(lambda_n)  # Differences between successive frequencies
    higher_order_diff = np.diff(delta_lambda_values)  # Second order difference
    delta_struct = np.max(higher_order_diff) / np.mean(delta_lambda_values)
    max_lambda_diff = np.max(np.diff(lambda_n))  # Maximum difference between frequencies
    return lambda_n[0], delta_struct, max_lambda_diff

# Run the tests for different parameters
for N in N_values:
    for C in C_values:
        for epsilon in epsilon_values:
            print(f"Testing for N={N}, C={C}, epsilon={epsilon}")

            # Check Conjecture 1: Structure Residual Boundedness
            conjecture_1_result = conjecture_1(x_range, N, C)
            print(f"Conjecture 1 (Residual boundedness): {conjecture_1_result}")

            # Check Conjecture 2: Collapse Entropy Attractor Principle
            min_entropy, min_x = conjecture_2(x_range, N, epsilon)
            print(f"Conjecture 2 (Entropy attractor principle): Minimum entropy = {min_entropy}, Occurs at x = {min_x}")

            # Check Conjecture 3: Spectral Mode Convergence Hierarchy
            lambda_0, delta_struct, max_lambda_diff = conjecture_3(x_range, N)
            print(f"Conjecture 3 (Spectral mode convergence): delta_struct = {delta_struct}, max_lambda_diff = {max_lambda_diff}")
            
            print("-" * 50)

# Plotting the results for one set of parameters
plt.plot(x_range, [delta(x, 50) for x in x_range], label="Residual Field")
plt.title('Residual Field vs x')
plt.xlabel('x')
plt.ylabel('δ(x)')
plt.show()
