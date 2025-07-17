import numpy as np
import matplotlib.pyplot as plt
import mpmath
from scipy.optimize import minimize, dual_annealing

# 设置 mpmath 精度
mpmath.mp.dps = 50

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
    x_mpf = mpmath.mpf(x)  # 将 x 转换为 mpmath 类型
    li_x = mpmath.li(x_mpf)  # 使用 mpmath 计算 Li(x)
    correction = mpmath.log(x_mpf) / mpmath.sqrt(x_mpf) + mpmath.power(mpmath.log(x_mpf), 2) / x_mpf + mpmath.power(mpmath.log(x_mpf), 3) / mpmath.sqrt(x_mpf)
    return float((li_x - correction) / x_mpf)

# 计算残差 delta(x)
def compute_delta(x, rho_x):
    pi_over_x = estimate_pi_over_x(x)
    return pi_over_x - rho_x

# 计算C值：残差最大值乘以log(x)
def compute_C(errors, x_values):
    max_error = max(errors)  # 获取最大误差
    min_x = min(x_values)  # 获取最小 x 值
    C_value = max_error * mpmath.log(min_x)  # 计算 C 值
    return float(C_value)  # 转换为 float 类型

# 优化 alpha, k, theta_n
def optimize_params_fine_tuned(x, lambda_n, N, initial_theta):
    def objective(params):
        alpha, k = params[:2]
        theta = params[2:]
        A_n = [k / ((n+1)**alpha) for n in range(N)]
        
        # 使用 float() 确保返回标量
        delta_value = compute_delta(x, compute_rho(x, lambda_n, A_n, theta, N))
        return float(abs(delta_value))  # 转换为标量

    # 更精细的初始猜测和边界
    initial_guess = [1.5, 1e-6] + [initial_theta[n] + np.random.uniform(-0.1, 0.1) for n in range(N)]
    bounds = [(0.5, 2), (1e-8, 1e-4)] + [(-np.pi, np.pi)]*N

    # 增加迭代次数
    result_lbfgs = min([minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000}) for _ in range(30)], key=lambda r: r.fun)
    result_global = dual_annealing(objective, bounds=bounds, maxiter=5000)
    
    # 返回最优结果
    return min([result_lbfgs, result_global], key=lambda r: objective(r.x))

# 主程序
x = mpmath.power(2, 2048)  # x 值设置为 2 的 2048 次方
N = 3  # 使用前三个零点
combinations = [
    [1, 2, 3],  # 低频
    [50, 75, 100],  # 中频
    [100, 125, 150],  # 高频
]

# 输出初步的基准值
pi_over_x = estimate_pi_over_x(x)
error_base = abs(1 / mpmath.log(x) - pi_over_x) / pi_over_x
print(f"Baseline: 1/log(x) error = {float(error_base):.6e}")

results = []
for combo in combinations:
    print(f"\n=== Testing Zero Combination {combo} ===")
    lambda_n = [rho_imag[n-1] for n in combo]
    initial_theta = [rho_arg[n-1] for n in combo]

    # 更精细优化
    result_fine_tuned = optimize_params_fine_tuned(x, lambda_n, N, initial_theta)
    alpha_opt, k_opt, theta_n_opt = result_fine_tuned.x[0], result_fine_tuned.x[1], result_fine_tuned.x[2:]
    A_n_opt = [k_opt / ((n+1)**alpha_opt) for n in range(N)]
    rho_x_opt = compute_rho(x, lambda_n, A_n_opt, theta_n_opt, N)
    delta_x_opt = compute_delta(x, rho_x_opt)
    error_rho_opt = abs(rho_x_opt - pi_over_x) / pi_over_x
    max_error = max([abs(compute_delta(x_val, compute_rho(x_val, lambda_n, A_n_opt, theta_n_opt, N))) for x_val in np.linspace(1e6, 1e8, 50)])
    
    # 计算C值
    C_value = compute_C([abs(compute_delta(x_val, compute_rho(x_val, lambda_n, A_n_opt, theta_n_opt, N))) for x_val in np.linspace(1e6, 1e8, 50)], np.linspace(1e6, 1e8, 50))

    # 存储结果
    results.append({
        "combo": combo,
        "C": C_value,
        "max_error": max_error,
        "delta_x_opt": delta_x_opt,
        "error_rho_opt": error_rho_opt,
        "improvement": float(error_base) / float(error_rho_opt),  # 转换为float避免mpf问题
        "k_opt": k_opt
    })
    print(f"Fine-Tuned Optimized: alpha = {alpha_opt:.6e}, k = {k_opt:.6e}")
    print(f"delta(x) = {float(delta_x_opt):.6e}")
    print(f"Relative error = {float(error_rho_opt):.6e}")
    print(f"Improvement over 1/log(x) = {float(error_base/error_rho_opt):.2f}x")
    print(f"C value = {C_value:.2e}")

# 打印结果
print("\n=== Results ===")
for result in results:
    print(f"Zero Point Combination: {result['combo']}")
    print(f"C value: {result['C']:.2e}")
    print(f"Prime density (pi(x)/x) at x = 2^2048: {float(pi_over_x):.6e}")
    print(f"Maximum error (small x): {float(result['max_error']):.2f}")
    print(f"Optimized delta(x) (large x): {float(result['delta_x_opt']):.6e}")
    print(f"Relative error after optimization: {float(result['error_rho_opt']):.6e}")
    print(f"Improvement factor: {float(result['improvement']):.2f}x")  # 添加float转换
    print(f"Optimized k: {result['k_opt']:.6e}\n")

# 可视化
x_values = np.linspace(1e6, 1e8, 50)
pi_over_x_values = [estimate_pi_over_x(x) for x in x_values]

plt.plot(x_values, pi_over_x_values, label='π(x)/x values')
plt.xlabel('x')
plt.ylabel('K(x) / π(x)/x')
plt.legend()
plt.grid(True)
plt.show()
