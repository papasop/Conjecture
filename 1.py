import numpy as np
import mpmath
from scipy.optimize import minimize, dual_annealing
import itertools
import random

# 设置 mpmath 精度
mpmath.mp.dps = 50

# 参数设置
x = mpmath.power(2, 2048)
N = 3
combinations = [
    [1, 2, 3],  # 低频
    [50, 75, 100],  # 中频
    [100, 125, 150],  # 高频
    [1, 50, 100],  # 混合
    [41, 109, 88], [72, 40, 56], [87, 27, 24]  # 随机组合
]

# 获取前 150 个零点虚部和相位
rho_imag = [mpmath.im(mpmath.zetazero(n+1)) for n in range(150)]
rho_arg = [mpmath.arg(mpmath.zetazero(n+1)) for n in range(150)]

# 计算 phi(x)
def compute_phi(x, lambda_n, A_n, theta_n, N):
    phi = mpmath.mpf(0)
    for n in range(N):
        phi += A_n[n] * mpmath.cos(lambda_n[n] * mpmath.log(x) + theta_n[n])
    return phi

# 计算 rho(x)
def compute_rho(x, lambda_n, A_n, theta_n, N):
    return 1 / mpmath.log(x) + compute_phi(x, lambda_n, A_n, theta_n, N)

# 估算 pi(x)/x
def estimate_pi_over_x(x):
    li_x = mpmath.li(x)
    correction = mpmath.log(x) / mpmath.sqrt(x) + mpmath.power(mpmath.log(x), 2) / x + mpmath.power(mpmath.log(x), 3) / mpmath.sqrt(x)
    return (li_x - correction) / x

# 计算残差 delta(x)
def compute_delta(x, rho_x):
    pi_over_x = estimate_pi_over_x(x)
    return pi_over_x - rho_x

# 优化 alpha, k, theta_n
def optimize_params(x, lambda_n, N, initial_theta):
    def objective(params):
        alpha, k = params[:2]
        theta = params[2:]
        A_n = [k / ((n+1)**alpha) for n in range(N)]
        return float(abs(compute_delta(x, compute_rho(x, lambda_n, A_n, theta, N))))
    initial_guess = [1.5, 1e-5] + [initial_theta[n] + np.random.uniform(-1, 1) for n in range(N)]
    bounds = [(0.5, 2), (1e-7, 1e-4)] + [(-np.pi, np.pi)]*N
    # 尝试 L-BFGS-B 和全局优化
    result_lbfgs = min([minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000}) for _ in range(20)], key=lambda r: r.fun)
    result_global = dual_annealing(objective, bounds=bounds, maxiter=2000)
    return min([result_lbfgs, result_global], key=lambda r: objective(r.x))

# 主程序
pi_over_x = estimate_pi_over_x(x)
error_base = abs(1 / mpmath.log(x) - pi_over_x) / pi_over_x
print(f"Baseline: 1/log(x) error = {float(error_base):.6e}")

for combo in combinations:
    print(f"\n=== Testing Zero Combination {combo} ===")
    lambda_n = [rho_imag[n-1] for n in combo]
    initial_theta = [rho_arg[n-1] for n in combo]
    
    # 未优化
    alpha = 1.5
    k = 1e-6
    A_n = [k / ((n+1)**alpha) for n in range(N)]
    rho_x = compute_rho(x, lambda_n, A_n, initial_theta, N)
    delta_x = compute_delta(x, rho_x)
    error_rho = abs(rho_x - pi_over_x) / pi_over_x
    print(f"Unoptimized: alpha = {alpha}, k = {k:.6e}")
    print(f"delta(x) = {float(delta_x):.6e}")
    print(f"Relative error = {float(error_rho):.6e}")
    print(f"Improvement over 1/log(x) = {float(error_base/error_rho):.2f}x")
    
    # 优化
    result = optimize_params(x, lambda_n, N, initial_theta)
    alpha_opt, k_opt, theta_n_opt = result.x[0], result.x[1], result.x[2:]
    A_n_opt = [k_opt / ((n+1)**alpha_opt) for n in range(N)]
    rho_x_opt = compute_rho(x, lambda_n, A_n_opt, theta_n_opt, N)
    delta_x_opt = compute_delta(x, rho_x_opt)
    error_rho_opt = abs(rho_x_opt - pi_over_x) / pi_over_x
    print(f"Optimized: alpha = {alpha_opt:.6e}, k = {k_opt:.6e}")
    print(f"delta(x) = {float(delta_x_opt):.6e}")
    print(f"Relative error = {float(error_rho_opt):.6e}")
    print(f"Improvement over 1/log(x) = {float(error_base/error_rho_opt):.2f}x")
