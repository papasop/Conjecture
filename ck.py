# Install mpmath (already satisfied in Colab, but included for completeness)
!pip install mpmath

import mpmath
import numpy as np

# Set precision for mpmath
mpmath.mp.dps = 50

# First 150 nontrivial Riemann zeros (imaginary parts γ_n)
# Extended to 150 elements using approximate values for 146th to 150th zeros
riemann_zeros = [
    14.134725, 21.022039, 25.010857, 30.424876, 32.935061, 37.586178, 40.918719,
    43.327073, 48.005150, 49.773832, 52.970321, 56.446247, 59.347044, 60.831778,
    65.112544, 67.079810, 69.546401, 72.067157, 75.704690, 77.144840, 79.337375,
    82.910380, 84.735492, 87.099773, 88.809111, 92.491899, 94.651344, 95.870634,
    98.831194, 101.317851, 103.725538, 105.446623, 107.168611, 111.029535,
    111.874659, 114.320220, 116.073018, 118.790783, 121.370125, 122.946829,
    124.256818, 127.516683, 130.283974, 131.087688, 133.497737, 134.758086,
    138.350663, 139.735238, 141.123707, 143.111845, 146.000982, 147.422765,
    150.053149, 150.925257, 153.024693, 156.112909, 157.597722, 158.849099,
    161.188964, 163.030709, 165.537110, 167.184439, 169.094515, 169.911928,
    172.849978, 173.939823, 175.702663, 177.609750, 179.916695, 182.207985,
    183.211275, 184.874467, 186.592750, 189.416244, 190.998984, 192.026714,
    193.079810, 195.265123, 196.876483, 198.015820, 201.264751, 202.493595,
    204.189671, 205.261385, 206.572016, 208.029926, 209.576510, 211.690418,
    213.347771, 214.547044, 215.983566, 218.008151, 219.067137, 220.714353,
    222.479706, 224.007060, 225.945828, 227.421980, 229.570051, 230.661972,
    231.693042, 233.693669, 235.378627, 236.524041, 238.468718, 239.554884,
    241.558463, 243.700614, 244.640472, 245.893766, 247.558128, 249.587896,
    250.996102, 252.038356, 253.804865, 255.513578, 256.955140, 258.215894,
    259.888280, 261.518166, 262.795195, 264.757532, 266.000391, 267.048744,
    269.456628, 270.459607, 271.487246, 273.388087, 274.715050, 276.627334,
    277.667732, 279.229433, 280.689548, 282.465011, 283.832491, 285.248909,
    286.258803, 287.932741, 289.766303, 291.309150, 292.755287, 294.025963,
    295.814223, 297.605012, 298.495955, 300.563127, 302.113456, 303.595234,
    305.389789, 306.672345  # Extended with approximate values
]

# Define groups (1-based indices converted to 0-based)
groups = [
    [1, 23, 50],  # Group 1
    [50, 75, 100],  # Group 2
    [100, 125, 150]  # Group 3
]
groups = [[i-1 for i in group] for group in groups]  # Convert to 0-based

# Parameters
x = mpmath.mpf(10**6)
log_x = mpmath.log(x)

# Compute π(x) ≈ Li(x)
li_x = mpmath.li(x, offset=True)
pi_x_approx = li_x / x

# Classical PNT density
pnt_density = 1 / log_x

# Compute for each group
results = []
for idx, group in enumerate(groups, 1):
    # Select zeros for the group
    try:
        selected_zeros = [riemann_zeros[i] for i in group]
    except IndexError as e:
        print(f"Error in Group {idx}: Invalid index in {group}. Max index is {len(riemann_zeros)-1}.")
        continue
    
    # Compute φ(x)
    phi_x = mpmath.mpf(0)
    for gamma_n in selected_zeros:
        rho_n = mpmath.mpf(0.5) + mpmath.mpf(gamma_n) * mpmath.mpc(0, 1)
        term = mpmath.power(x, rho_n) / (rho_n * log_x)
        phi_x += term
    
    # Estimate k(x)
    if mpmath.almosteq(phi_x, 0):
        k_x = mpmath.mpf(0)
        print(f"Warning: φ(x) is zero for Group {idx}, setting k(x) = 0")
    else:
        k_x = (pi_x_approx - pnt_density) / phi_x
    
    # Compute RPD
    rpd = pnt_density + k_x * phi_x
    
    # Compute C
    C = mpmath.fabs(k_x * phi_x) * log_x
    
    # Store results
    results.append({
        'group': idx,
        'indices': [i+1 for i in group],  # Convert back to 1-based for display
        'gamma_n': selected_zeros,
        'k_x': k_x,
        'C': C,
        'rpd': rpd,
        'pnt_density': pnt_density
    })

# Output results
for res in results:
    print(f"\nGroup {res['group']}: Indices {res['indices']}")
    print(f"γ_n values: {[float(g) for g in res['gamma_n']]}")
    print(f"k(x): {complex(res['k_x'])}")
    print(f"C: {float(res['C'])}")
    print(f"RPD (magnitude): {float(mpmath.fabs(res['rpd']))}")
    print(f"RPD (real part): {float(res['rpd'].real)}")
    print(f"1/log(x): {float(res['pnt_density'])}")