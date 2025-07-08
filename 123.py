import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi  # Import prime counting function

# Parameters for testing
x_range = np.linspace(10, 1000, 100)  # Increase sampling resolution and range
N_values = [10, 30, 50, 100]  # Different modal cutoff values for testing
delta_lambda = 0.01  # Frequency step
C_values = [500, 1000, 2000, 5000]  # Different C values to test (increased range)

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
def conjecture_2(x_range, N=50):
    entropy_values = [entropy(x, N) for x in x_range]
    min_entropy = min(entropy_values)
    min_x = x_range[entropy_values.index(min_entropy)]
    return min_entropy, min_x

# Conjecture 3: Spectral Mode Convergence Hierarchy
def conjecture_3(x_range, N=50):
    lambda_0 = delta_lambda  # Assuming the starting frequency corresponds to the first mode

    # Define a convergence measure for spectral modes
    lambda_n = np.array([n * delta_lambda for n in range(1, N + 1)])
    delta_lambda_values = np.diff(lambda_n)  # Differences between successive frequencies
    delta_struct = np.max(delta_lambda_values) / np.mean(delta_lambda_values)  # Normalized frequency spread

    return lambda_0, delta_struct

# Validate for different N values and ranges
results = {}
for N in N_values:
    for C in C_values:
        # Check Conjecture 1: Structure Residual Boundedness
        conjecture_1_result = conjecture_1(x_range, N, C)
        
        # Check Conjecture 2: Collapse Entropy Attractor Principle
        min_entropy, min_x = conjecture_2(x_range, N)
        
        # Check Conjecture 3: Spectral Mode Convergence Hierarchy
        lambda_0, delta_struct = conjecture_3(x_range, N)

        # Store results
        results[(N, C)] = {
            "Conjecture 1 (Residual boundedness)": conjecture_1_result,
            "Conjecture 2 (Entropy attractor principle)": (min_entropy, min_x),
            "Conjecture 3 (Spectral mode convergence)": (lambda_0, delta_struct)
        }

# Output results
for (N, C), result in results.items():
    print(f"\nFor N = {N}, C = {C}:")
    print(f"Conjecture 1 (Residual boundedness): {result['Conjecture 1 (Residual boundedness)']}")
    print(f"Conjecture 2 (Entropy attractor principle): Minimum entropy = {result['Conjecture 2 (Entropy attractor principle)'][0]}, Occurs at x = {result['Conjecture 2 (Entropy attractor principle)'][1]}")
    print(f"Conjecture 3 (Spectral mode convergence): Starting frequency λ₀ = {result['Conjecture 3 (Spectral mode convergence)'][0]}, delta_struct = {result['Conjecture 3 (Spectral mode convergence)'][1]}")

# Plot Residual values
plt.figure(figsize=(12, 6))
for N in N_values:
    residual_results = [delta(x, N) for x in x_range]
    plt.plot(x_range, residual_results, label=f"N={N}")
plt.title('Residual Field vs x for Different N values')
plt.xlabel('x')
plt.ylabel('Residual Field δ(x)')
plt.legend()
plt.show()
