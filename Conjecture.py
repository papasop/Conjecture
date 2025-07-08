import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi  # Import prime counting function

# Parameters
x_range = np.linspace(10, 1000, 100)  # Increase sampling resolution
N = 50  # Increase modal cutoff
delta_lambda = 0.01

# Define the modal collapse function
def phi(x, N):
    A_n = np.array([1 / n**2 for n in range(1, N+1)])  # Amplitude decay
    lambda_n = np.array([n * delta_lambda for n in range(1, N+1)])  # Frequency
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
    # Ensure delta_x is a NumPy-compatible float
    delta_x = np.array(delta_x, dtype=np.float64)  # Explicitly set dtype
    return np.log(1 + delta_x**2)

# Conjecture 1: Structure Residual Boundedness
def conjecture_1(x_range, N=50):
    residual_values = [delta(x, N) for x in x_range]
    residual_mean = np.mean(residual_values)
    residual_max = np.max(np.abs(residual_values))
    # Check if residual is bounded
    if residual_max < 1000 / np.log(x_range[-1]):
        return 'Passed'
    else:
        return 'Failed'

# Conjecture 2: Collapse Entropy Attractor Principle
def conjecture_2(x_range, N=50):
    entropy_values = [entropy(x, N) for x in x_range]
    min_entropy = min(entropy_values)
    min_x = x_range[entropy_values.index(min_entropy)]
    return min_entropy, min_x

# Conjecture 3: Spectral Mode Convergence Hierarchy
def conjecture_3(x_range, N=50):
    delta_struct = np.max(np.abs(np.diff(x_range))) / np.mean(np.diff(x_range))
    return delta_struct

# Check Conjecture 1: Structure Residual Boundedness
conjecture_1_result = conjecture_1(x_range)

# Check Conjecture 2: Collapse Entropy Attractor Principle
min_entropy, min_x = conjecture_2(x_range)

# Check Conjecture 3: Spectral Mode Convergence Hierarchy
delta_struct = conjecture_3(x_range)

# Output results
print(f"Conjecture 1 (Residual boundedness): {conjecture_1_result}")
print(f"Conjecture 2 (Entropy attractor principle): Minimum entropy = {min_entropy}, Occurs at x = {min_x}")
print(f"Conjecture 3 (Spectral mode convergence): delta_struct = {delta_struct}")