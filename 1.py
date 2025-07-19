import numpy as np
import mpmath
from scipy.optimize import minimize, dual_annealing
import matplotlib.pyplot as plt
from mpmath import zetazero, log

# 设置 mpmath 精度
mpmath.mp.dps = 50

# 计算参考素数密度 1/log(x)
def prime_density_ref(x):
    return 1 / log(x)

# 计算结构密度函数（3-mode模型）
def rho_structured(x, amps, phases, freqs):
    terms = [a * np.cos(f * np.log(x) + p) for a, p, f in zip(amps, phases, freqs)]
    return np.sum(terms, axis=0)

# 计算残差 delta(x)
def delta(x, amps, phases, freqs):
    return rho_structured(x, amps, phases, freqs) - prime_density_ref(x)

# 计算相对误差
def relative_error(x, amps, phases, freqs):
    return abs(delta(x, amps, phases, freqs)) / prime_density_ref(x)

# 优化的目标函数：最小化残差
def loss(params, x, freqs):
    amps = params[:3]
    phases = params[3:]
    return float(abs(delta(x, amps, phases, freqs)))

# 主测试函数
def test_combination(zero_ids, x_eval=1000):
    freqs = [float(mpmath.im(zetazero(n))) for n in zero_ids]

    # 初始参数
    init = np.array([1.0] * 3 + [0.0] * 3)

    # 优化
    res = minimize(loss, init, args=(x_eval, freqs), method='Nelder-Mead')
    amps_opt = res.x[:3]
    phases_opt = res.x[3:]
    
    # 计算优化后的结构密度
    rho_opt = rho_structured(x_eval, amps_opt, phases_opt, freqs)
    delta_opt = delta(x_eval, amps_opt, phases_opt, freqs)
    rel_error_opt = relative_error(x_eval, amps_opt, phases_opt, freqs)

    # 计算参考素数密度
    ref_density = prime_density_ref(x_eval)

    return {
        "zeros": zero_ids,
        "amps_opt": amps_opt,
        "phases_opt": phases_opt,
        "max_residual_c": float(abs(delta_opt)),  # 转换为普通浮点数
        "rel_error_opt": float(rel_error_opt),  # 转换为普通浮点数
        "ref_density": float(ref_density),  # 转换为普通浮点数
        "structured_density": float(rho_opt),  # 转换为普通浮点数
        "logx_density": float(ref_density),  # 转换为普通浮点数
    }

# 测试的零点组合
zero_combos = [
    [1, 2, 75],
    [1, 2, 100],
    [1, 2, 125],
    [1, 50, 75],
]

# 运行测试
results = [test_combination(combo) for combo in zero_combos]

# 输出结果
for result in results:
    print(f"=== Zero Combination {result['zeros']} ===")
    print(f"Max Residual c: {result['max_residual_c']:.6e}")
    print(f"1/log(x): {result['logx_density']:.6e}")
    print(f"Structured Density: {result['structured_density']:.6e}")
    print(f"Relative Error: {result['rel_error_opt']:.6e}")
    print("-" * 50)
