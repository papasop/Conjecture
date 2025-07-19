from mpmath import mp, mpf, pi, tan, log, fsum, atan
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- High precision ---
mp.dps = 50

# --- Machin-like correction term ϕ ---
def compute_phi():
    a_k = [4, -1]
    c_k = [5, 239]
    phi = mpf(0)
    for a, c in zip(a_k, c_k):
        phi += a * atan(mpf(1) / mpf(c))
    return phi

phi_fixed = compute_phi()

# --- α(n) = 4 / (π·n) ---
def alpha(n):
    return mpf(4) / (mp.pi * mpf(n))

# --- Leibniz partial sum ---
def leibniz_sum(n):
    return fsum([mpf(4) * (-1)**k / (2*k + 1) for k in range(n)])

# --- Structured π approximation ---
def pi_structured(n):
    return leibniz_sum(n) + alpha(n) * phi_fixed

# --- K(t) function ---
def K(t):
    return -mp.pi * tan(mp.pi * t) / log(mp.pi)

# --- Compute entropy of residuals or K deviations ---
def compute_entropy(values):
    values = np.array([float(v) for v in values])
    values = values / np.sum(values)  # Normalize
    values = values[values > 0]
    entropy = -np.sum(values * np.log2(values))
    return entropy

# --- Test K(t) under ε ≈ 1.99999 with corrected mapping ---
def test_K_under_epsilon_corrected():
    ns = [100, 1000, 10000, 100000, 1000000]
    t_base = mpf(0.888774)  # Attracting point from second paper
    c = mpf(0.01)  # Small constant for mapping
    true_pi = mp.pi
    residuals = []
    
    print("🎯 Testing K(t) under ε ≈ 1.99999 with t = 0.888774 + c/n")
    print("="*70)
    
    for n in ns:
        approx = pi_structured(n)
        residual = abs(true_pi - approx)
        residuals.append(residual)
        
        t = t_base + c / mpf(n)
        k_val = K(t)
        
        print(f"n = {n:<8} | t = {mp.nstr(t, 20):<20} | K(t) = {mp.nstr(k_val, 20)}")
        print(f"         | ρ(n) ≈ {mp.nstr(approx, 20)}")
        print(f"         | Residual = {mp.nstr(residual, 20)}")
        print(f"         | Residual × n^3 = {mp.nstr(residual * n**3, 20)}")
        print("-"*70)
    
    # Fit log-log to verify ε
    log_n = np.log(np.array(ns, dtype=np.float64))
    log_res = np.log(np.array([float(r) for r in residuals]))
    slope, intercept = np.polyfit(log_n, log_res, 1)
    epsilon_est = -slope - 1
    print(f"\n✅ Estimated ε ≈ {epsilon_est:.5f} (target: ε > 0)")
    
    # Compute entropy
    entropy_residuals = compute_entropy(residuals)
    print(f"Entropy of residuals = {entropy_residuals:.5f} bits")
    
    # Test K(t) at t ≈ 0.888774
    k_target = K(t_base)
    print(f"K(t ≈ 0.888774) = {mp.nstr(k_target, 20)}")
    
    # Entropy of |K(t) - 1|
    t_values = np.linspace(0.5, 1.0, 1000)
    K_deviations = [abs(K(mpf(t)) - 1) for t in t_values]
    entropy_K = compute_entropy(K_deviations)
    print(f"Entropy of |K(t) - 1| = {entropy_K:.5f} bits")
    
    # Plot
    t_vals = [t_base + c / mpf(n) for n in ns]
    K_vals = [K(t) for t in t_vals]
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [float(k) for k in K_vals], marker='o', label='K(t) with t = 0.888774 + c/n')
    plt.axhline(y=1, color='r', linestyle='--', label='K = 1')
    plt.xscale('log')
    plt.title("K(t) vs n with t = 0.888774 + c/n (c = 0.01)")
    plt.xlabel("n (log scale)")
    plt.ylabel("K(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

# --- Run ---
test_K_under_epsilon_corrected()