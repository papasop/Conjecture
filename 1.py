import numpy as np
import mpmath
from scipy.optimize import minimize
from sympy import primepi

# 设置 mpmath 精度
mpmath.mp.dps = 50

# 参数设置
N = 100  # 测试 N=100
alpha_values = [1, 2, 3]
k_values = [1e-7, 1e-8]  # 测试 k=10^-7, 10^-8
x_values = [mpmath.power(2, 1024), mpmath.power(2, 2048), mpmath.mpf(1e9), mpmath.mpf(1e10)]

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

# 估算 pi(x)/x（使用真实 pi(x) 或 li(x)/x）
def estimate_pi_over_x(x):
    x_int = int(mpmath.floor(x))
    if x_int <= 10**10:  # 使用 sympy 计算真实 pi(x)
        pi_x = primepi(x_int)
        return mpmath.mpf(pi_x) / x
    else:  # 使用改进 li(x)/x
        li_x = mpmath.li(x)
        correction = mpmath.log(x) / mpmath.sqrt(x)  # 次阶项
        return (li_x - correction) / x

# 计算残差 delta(x)
def compute_delta(x, rho_x):
    pi_over_x = estimate_pi_over_x(x)
    return pi_over_x - rho_x

# 验证 |delta(x)| < C/log(x)
def verify_bound(delta, x, C):
    bound = C / mpmath.log(x)
    return abs(delta) < bound, bound

# 优化 k
def optimize_k(x, lambda_n, theta_n, N, alpha, k0=1e-6):
    def objective(k):
        A_n = [k[0] / ((n+1)**alpha) for n in range(N)]
        rho_x = compute_rho(x, lambda_n, A_n, theta_n, N)
        return float(abs(compute_delta(x, rho_x)))
    result = minimize(objective, x0=[k0], method='Nelder-Mead')
    return result.x[0]

# 优化 theta_n
def optimize_theta_n(x, lambda_n, A_n, N):
    def objective(theta):
        rho_x = compute_rho(x, lambda_n, A_n, theta, N)
        return float(abs(compute_delta(x, rho_x)))
    result = minimize(objective, x0=[0]*N, method='Nelder-Mead')
    return result.x

# 主程序
print(f"\n=== Testing N = {N} ===")
for alpha in alpha_values:
    print(f"\n=== Testing alpha = {alpha} ===")
    
    # 设置频率 lambda_n = Im(rho_n)
    lambda_n = rho_imag[:N]
    
    for k in k_values:
        print(f"\n=== Testing k = {k} ===")
        
        # 生成振幅 A_n = k/n^alpha
        A_n = [k / ((n+1)**alpha) for n in range(N)]
        
        # 使用零点相位
        theta_n = rho_arg[:N]
        
        for x in x_values:
            x_str = f"10^{int(mpmath.log10(x))}" if x <= 1e10 else f"2^{int(mpmath.log(x)/mpmath.log(2))}"
            print(f"\nTesting x = {x_str}")
            
            # 计算 phi(x) 和 rho(x)
            phi_x = compute_phi(x, lambda_n, A_n, theta_n, N)
            rho_x = compute_rho(x, lambda_n, A_n, theta_n, N)
            
            # 计算 delta(x)
            delta_x = compute_delta(x, rho_x)
            
            # 估算 pi(x)/x
            pi_over_x = estimate_pi_over_x(x)
            
            # 验证界限
            C_values = [10, 100, 1000]
            bound_results = []
            for C in C_values:
                is_bounded, bound = verify_bound(delta_x, x, C)
                bound_results.append((C, is_bounded, bound))
            
            # 比较效率
            error_rho = abs(rho_x - pi_over_x) / pi_over_x
            error_base = abs(1 / mpmath.log(x) - pi_over_x) / pi_over_x
            
            # 输出结果
            print(f"phi(x) = {float(phi_x):.6e}")
            print(f"rho(x) = 1/log(x) + phi(x) = {float(rho_x):.6e}")
            print(f"pi(x)/x (approx) = {float(pi_over_x):.6e}")
            print(f"delta(x) = {float(delta_x):.6e}")
            print("\nBound verification (|delta(x)| < C/log(x)):")
            for C, is_bounded, bound in bound_results:
                print(f"C = {C}: |delta(x)| = {float(abs(delta_x)):.6e} < {float(bound):.6e} ? {is_bounded}")
            print(f"\nRelative error of rho(x): {float(error_rho):.6e}")
            print(f"Relative error of 1/log(x): {float(error_base):.6e}")
            
            # 优化 k
            print(f"\nOptimizing k for x = {x_str}")
            k_opt = optimize_k(x, lambda_n, theta_n, N, alpha)
            A_n_opt = [k_opt / ((n+1)**alpha) for n in range(N)]
            rho_x_opt = compute_rho(x, lambda_n, A_n_opt, theta_n, N)
            delta_x_opt = compute_delta(x, rho_x_opt)
            error_rho_opt = abs(rho_x_opt - pi_over_x) / pi_over_x
            print(f"Optimized k = {k_opt:.6e}")
            print(f"Optimized delta(x) = {float(delta_x_opt):.6e}")
            print(f"Optimized relative error of rho(x): {float(error_rho_opt):.6e}")
            
            # 优化 theta_n（仅对 x=10^9 测试，因计算成本高）
            if x <= 1e9:
                print(f"\nOptimizing theta_n for x = {x_str}")
                theta_n_opt = optimize_theta_n(x, lambda_n, A_n, N)
                rho_x_opt_theta = compute_rho(x, lambda_n, A_n, theta_n_opt, N)
                delta_x_opt_theta = compute_delta(x, rho_x_opt_theta)
                error_rho_opt_theta = abs(rho_x_opt_theta - pi_over_x) / pi_over_x
                print(f"Optimized delta(x) with theta_n = {float(delta_x_opt_theta):.6e}")
                print(f"Optimized relative error of rho(x) with theta_n: {float(error_rho_opt_theta):.6e}")