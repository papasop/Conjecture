{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import matplotlib.pyplot as plt\
import mpmath\
from scipy.optimize import minimize, dual_annealing\
\
# \uc0\u35774 \u32622  mpmath \u31934 \u24230 \
mpmath.mp.dps = 50\
\
# \uc0\u33719 \u21462 \u21069  150 \u20010 \u38646 \u28857 \u34394 \u37096 \u21644 \u30456 \u20301 \
rho_imag = [mpmath.im(mpmath.zetazero(n+1)) for n in range(150)]\
rho_arg = [mpmath.arg(mpmath.zetazero(n+1)) for n in range(150)]\
\
# \uc0\u35745 \u31639  phi(x)\
def compute_phi(x, lambda_n, A_n, theta_n, N):\
    phi = mpmath.mpf(0)\
    for n in range(N):\
        phi += A_n[n] * mpmath.cos(lambda_n[n] * mpmath.log(x) + theta_n[n])\
    return phi\
\
# \uc0\u35745 \u31639  rho(x)\
def compute_rho(x, lambda_n, A_n, theta_n, N):\
    return 1 / mpmath.log(x) + compute_phi(x, lambda_n, A_n, theta_n, N)\
\
# \uc0\u20272 \u31639  pi(x)/x\
def estimate_pi_over_x(x):\
    x_mpf = mpmath.mpf(x)  # \uc0\u23558  x \u36716 \u25442 \u20026  mpmath \u31867 \u22411 \
    li_x = mpmath.li(x_mpf)  # \uc0\u20351 \u29992  mpmath \u35745 \u31639  Li(x)\
    correction = mpmath.log(x_mpf) / mpmath.sqrt(x_mpf) + mpmath.power(mpmath.log(x_mpf), 2) / x_mpf + mpmath.power(mpmath.log(x_mpf), 3) / mpmath.sqrt(x_mpf)\
    return float((li_x - correction) / x_mpf)\
\
# \uc0\u35745 \u31639 \u27531 \u24046  delta(x)\
def compute_delta(x, rho_x):\
    pi_over_x = estimate_pi_over_x(x)\
    return pi_over_x - rho_x\
\
# \uc0\u35745 \u31639 C\u20540 \u65306 \u27531 \u24046 \u26368 \u22823 \u20540 \u20056 \u20197 log(x)\
def compute_C(errors, x_values):\
    max_error = max(errors)  # \uc0\u33719 \u21462 \u26368 \u22823 \u35823 \u24046 \
    min_x = min(x_values)  # \uc0\u33719 \u21462 \u26368 \u23567  x \u20540 \
    C_value = max_error * mpmath.log(min_x)  # \uc0\u35745 \u31639  C \u20540 \
    return float(C_value)  # \uc0\u36716 \u25442 \u20026  float \u31867 \u22411 \
\
# \uc0\u20248 \u21270  alpha, k, theta_n\
def optimize_params_fine_tuned(x, lambda_n, N, initial_theta):\
    def objective(params):\
        alpha, k = params[:2]\
        theta = params[2:]\
        A_n = [k / ((n+1)**alpha) for n in range(N)]\
        \
        # \uc0\u20351 \u29992  float() \u30830 \u20445 \u36820 \u22238 \u26631 \u37327 \
        delta_value = compute_delta(x, compute_rho(x, lambda_n, A_n, theta, N))\
        return float(abs(delta_value))  # \uc0\u36716 \u25442 \u20026 \u26631 \u37327 \
\
    # \uc0\u26356 \u31934 \u32454 \u30340 \u21021 \u22987 \u29468 \u27979 \u21644 \u36793 \u30028 \
    initial_guess = [1.5, 1e-6] + [initial_theta[n] + np.random.uniform(-0.1, 0.1) for n in range(N)]\
    bounds = [(0.5, 2), (1e-8, 1e-4)] + [(-np.pi, np.pi)]*N\
\
    # \uc0\u22686 \u21152 \u36845 \u20195 \u27425 \u25968 \
    result_lbfgs = min([minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=bounds, options=\{'maxiter': 5000\}) for _ in range(30)], key=lambda r: r.fun)\
    result_global = dual_annealing(objective, bounds=bounds, maxiter=5000)\
    \
    # \uc0\u36820 \u22238 \u26368 \u20248 \u32467 \u26524 \
    return min([result_lbfgs, result_global], key=lambda r: objective(r.x))\
\
# \uc0\u20027 \u31243 \u24207 \
x = mpmath.power(2, 2048)  # x \uc0\u20540 \u35774 \u32622 \u20026  2 \u30340  2048 \u27425 \u26041 \
N = 3  # \uc0\u20351 \u29992 \u21069 \u19977 \u20010 \u38646 \u28857 \
combinations = [\
    [1, 2, 3],  # \uc0\u20302 \u39057 \
    [50, 75, 100],  # \uc0\u20013 \u39057 \
    [100, 125, 150],  # \uc0\u39640 \u39057 \
]\
\
# \uc0\u36755 \u20986 \u21021 \u27493 \u30340 \u22522 \u20934 \u20540 \
pi_over_x = estimate_pi_over_x(x)\
error_base = abs(1 / mpmath.log(x) - pi_over_x) / pi_over_x\
print(f"Baseline: 1/log(x) error = \{float(error_base):.6e\}")\
\
results = []\
for combo in combinations:\
    print(f"\\n=== Testing Zero Combination \{combo\} ===")\
    lambda_n = [rho_imag[n-1] for n in combo]\
    initial_theta = [rho_arg[n-1] for n in combo]\
\
    # \uc0\u26356 \u31934 \u32454 \u20248 \u21270 \
    result_fine_tuned = optimize_params_fine_tuned(x, lambda_n, N, initial_theta)\
    alpha_opt, k_opt, theta_n_opt = result_fine_tuned.x[0], result_fine_tuned.x[1], result_fine_tuned.x[2:]\
    A_n_opt = [k_opt / ((n+1)**alpha_opt) for n in range(N)]\
    rho_x_opt = compute_rho(x, lambda_n, A_n_opt, theta_n_opt, N)\
    delta_x_opt = compute_delta(x, rho_x_opt)\
    error_rho_opt = abs(rho_x_opt - pi_over_x) / pi_over_x\
    max_error = max([abs(compute_delta(x_val, compute_rho(x_val, lambda_n, A_n_opt, theta_n_opt, N))) for x_val in np.linspace(1e6, 1e8, 50)])\
    \
    # \uc0\u35745 \u31639 C\u20540 \
    C_value = compute_C([abs(compute_delta(x_val, compute_rho(x_val, lambda_n, A_n_opt, theta_n_opt, N))) for x_val in np.linspace(1e6, 1e8, 50)], np.linspace(1e6, 1e8, 50))\
\
    # \uc0\u23384 \u20648 \u32467 \u26524 \
    results.append(\{\
        "combo": combo,\
        "C": C_value,\
        "max_error": max_error,\
        "delta_x_opt": delta_x_opt,\
        "error_rho_opt": error_rho_opt,\
        "improvement": float(error_base) / float(error_rho_opt),  # \uc0\u36716 \u25442 \u20026 float\u36991 \u20813 mpf\u38382 \u39064 \
        "k_opt": k_opt\
    \})\
    print(f"Fine-Tuned Optimized: alpha = \{alpha_opt:.6e\}, k = \{k_opt:.6e\}")\
    print(f"delta(x) = \{float(delta_x_opt):.6e\}")\
    print(f"Relative error = \{float(error_rho_opt):.6e\}")\
    print(f"Improvement over 1/log(x) = \{float(error_base/error_rho_opt):.2f\}x")\
    print(f"C value = \{C_value:.2e\}")\
\
# \uc0\u25171 \u21360 \u32467 \u26524 \
print("\\n=== Results ===")\
for result in results:\
    print(f"Zero Point Combination: \{result['combo']\}")\
    print(f"C value: \{result['C']:.2e\}")\
    print(f"Prime density (pi(x)/x) at x = 2^2048: \{float(pi_over_x):.6e\}")\
    print(f"Maximum error (small x): \{float(result['max_error']):.2f\}")\
    print(f"Optimized delta(x) (large x): \{float(result['delta_x_opt']):.6e\}")\
    print(f"Relative error after optimization: \{float(result['error_rho_opt']):.6e\}")\
    print(f"Improvement factor: \{float(result['improvement']):.2f\}x")  # \uc0\u28155 \u21152 float\u36716 \u25442 \
    print(f"Optimized k: \{result['k_opt']:.6e\}\\n")\
\
# \uc0\u21487 \u35270 \u21270 \
x_values = np.linspace(1e6, 1e8, 50)\
pi_over_x_values = [estimate_pi_over_x(x) for x in x_values]\
\
plt.plot(x_values, pi_over_x_values, label='\uc0\u960 (x)/x values')\
plt.xlabel('x')\
plt.ylabel('K(x) / \uc0\u960 (x)/x')\
plt.legend()\
plt.grid(True)\
plt.show()\
}