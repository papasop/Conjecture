import numpy as np
import mpmath
from scipy.optimize import minimize
from sympy import primepi

# 设置 mpmath 精度
mpmath.mp.dps = 50

# 参数设置
N = 100
alpha_values = [1, 2, 3]
k = 1e-7
x_values = [mpmath.power(2, 2048)]

# 获取前 100 个零点虚部和相位
rho_imag = [mpmath.im(mpmath.zetazero(n+1)) for n in range(N)]
rho_arg = [mpmath.arg(mpmath.zetazero(n+1)) for n in range(N)]

# 计算 phi(x)
def compute_phi(x, lambda_n, A_n, theta_n, N):
    phi = mpmath.mpf(0)
    for n in range(N):
        phi += A_n[n] * mpmath.cos(lambda_n[n] * mpmath.log(x) + theta_n[n])
    return phi

# 计算 rho(x) = 1/log(x) + phi(x)
def compute_rho(x, lambda_n, A_n, theta_n, N):
    return 1 / mpmath.log(x) + compute_phi(x, lambda_n, A_n, theta_n, N)

# 估算 pi(x)/x
def estimate_pi_over_x(x):
    li_x = mpmath.li(x)
    correction = mpmath.log(x) / mpmath.sqrt(x) + mpmath.power(mpmath.log(x), 2) / x
    return (li_x - correction) / x

# 计算残差 delta(x)
def compute_delta(x, rho_x):
    pi_over_x = estimate_pi_over_x(x)
    return pi_over_x - rho_x

# 验证 |delta(x)| < C/log(x)
def verify_bound(delta, x, C):
    bound = C / mpmath.log(x)
    return abs(delta) < bound, bound

# 优化 theta_n（降低维度，M=50）
def optimize_theta_n(x, lambda_n, A_n, N, M=50):
    def objective(theta):
        full_theta = list(theta) + list(rho_arg[M:N])
        rho_x = compute_rho(x, lambda_n, A_n, full_theta, N)
        return float(abs(compute_delta(x, rho_x))) + 1e-6 * np.sum(np.square(full_theta))
    initial_theta = [float(rho_arg[n]) + np.random.uniform(-0.1, 0.1) for n in range(M)]
    result = minimize(objective, x0=initial_theta, method='L-BFGS-B', bounds=[(-np.pi, np.pi)]*M, options={'maxiter': 1000})
    return list(result.x) + list(rho_arg[M:N])

# 主程序
print(f"\n=== Testing N = {N} ===")
for alpha in alpha_values:
    print(f"\n=== Testing alpha = {alpha} ===")
    
    A_n = [k / ((n+1)**alpha) for n in range(N)]
    lambda_n = rho_imag[:N]
    
    for x in x_values:
        x_str = f"2^{int(mpmath.log(x)/mpmath.log(2))}"
        print(f"\nTesting x = {x_str}")
        
        theta_n = rho_arg[:N]
        phi_x = compute_phi(x, lambda_n, A_n, theta_n, N)
        rho_x = compute_rho(x, lambda_n, A_n, theta_n, N)
        delta_x = compute_delta(x, rho_x)
        pi_over_x = estimate_pi_over_x(x)
        
        C_values = [10, 100, 1000]
        bound_results = []
        for C in C_values:
            is_bounded, bound = verify_bound(delta_x, x, C)
            bound_results.append((C, is_bounded, bound))
        
        error_rho = abs(rho_x - pi_over_x) / pi_over_x
        error_base = abs(1 / mpmath.log(x) - pi_over_x) / pi_over_x
        
        print(f"phi(x) = {float(phi_x):.6e}")
        print(f"rho(x) = 1/log(x) + phi(x) = {float(rho_x):.6e}")
        print(f"pi(x)/x (approx) = {float(pi_over_x):.6e}")
        print(f"delta(x) = {float(delta_x):.6e}")
        print("\nBound verification (|delta(x)| < C/log(x)):")
        for C, is_bounded, bound in bound_results:
            print(f"C = {C}: |delta(x)| = {float(abs(delta_x)):.6e} < {float(bound):.6e} ? {is_bounded}")
        print(f"\nRelative error of rho(x): {float(error_rho):.6e}")
        print(f"Relative error of 1/log(x): {float(error_base):.6e}")
        
        print(f"\nOptimizing theta_n for x = {x_str}")
        theta_n_opt = optimize_theta_n(x, lambda_n, A_n, N)
        rho_x_opt = compute_rho(x, lambda_n, A_n, theta_n_opt, N)
        delta_x_opt = compute_delta(x, rho_x_opt)
        error_rho_opt = abs(rho_x_opt - pi_over_x) / pi_over_x
        print(f"Optimized delta(x) = {float(delta_x_opt):.6e}")
        print(f"Optimized relative error of rho(x): {float(error_rho_opt):.6e}")