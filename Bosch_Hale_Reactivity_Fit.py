import numpy as np
import matplotlib.pyplot as plt

# Bosch Hale Reactivity Fit

# Bosch-Hale Fit Parameters from Table VII
reactivity_fits = {
    "T(d,n)4He": {
        "B_G": 34.3827,
        "m_rc2": 1124656,
        "C": [1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5],
        "Ti_range_keV": [0.2, 100],
        "max_dev_percent": 0.25
    },
    "D(d,n)3He": {
        "B_G": 31.3970,
        "m_rc2": 937814,
        "C": [5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0, -2.96400e-6, 0.0, 0.0],
        "Ti_range_keV": [0.2, 100],
        "max_dev_percent": 0.3
    },
    #  "3He(d,p)4He": {
    #     "B_G": 68.7508,
    #     "m_rc2": 1124572,
    #     "C": [5.51036e-10, 6.41918e-3, -2.02896e-3, -1.91080e-5, 1.35776e-4, 0.0, 0.0],
    #    "Ti_range_keV": [0.5, 190],
    #     "max_dev_percent": 2.5
    # },

}

color_map = {
    "T(d,n)4He": "blue",
    "D(d,n)3He": "green",
    "3He(d,p)4He": "yellow"
}

### Reactivity computation functions

def compute_theta(T, C):
    """Equation 13: Compute theta(T) using fit coefficients C."""
    num = T
    den = 1 - (T * (C[1] + T * (C[3] + T * C[5]))) / \
        (1 + T * (C[2] + T * (C[4] + T * C[6])))
    
    return num / den

def compute_xi(B_G, theta_val):
    """Equation 14: Compute xi from theta and B_G."""
    return (B_G**2 / (4 * theta_val))**(1 / 3)

def sigma_v(T, fit):
    """Equation 12: Calculate ⟨σv⟩ for a given temperature T and reactivity fit parameters."""
    C = fit["C"]
    B_G = fit["B_G"]
    m_rc2 = fit["m_rc2"]  # keV
    theta_val = compute_theta(T, C)
    xi_val = compute_xi(B_G, theta_val)
    sqrt_term = np.sqrt(xi_val / (m_rc2 * T**3))
    return C[0] * theta_val * sqrt_term * np.exp(-3 * xi_val)  # cm^3/s

# Plotting Data
plt.figure(figsize=(10, 6))

for reaction, fit in reactivity_fits.items():
    T_min, T_max = fit["Ti_range_keV"]
    T_vals = np.linspace(T_min, T_max, 500)
    sv_vals = [sigma_v(T, fit) for T in T_vals]
    plt.plot(T_vals, sv_vals, label=reaction, color=color_map[reaction])

plt.yscale("log")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel("Ion Temperature T (keV)", fontsize=12)
plt.ylabel("Fusion Reactivity ⟨σv⟩ (cm³/s)", fontsize=12)
plt.title("Fusion Reactivity (Bosch-Hale Fits)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------
# Validity Check with Bosch Hale Tabulated Values from Table VIII

BH_tableVIII = {
    "Ti (keV)": [
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5,
        3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0
    ],
    r"D(t,n)\alpha Reaction Rate ($cm^{-3} s^{-1}$)": [
        1.254e-26, 7.292e-25, 9.344e-24, 5.697e-23, 2.253e-22, 
        6.740e-22, 1.662e-21, 6.857e-21, 2.546e-20, 6.923e-20, 
        1.539e-19, 2.977e-19, 8.425e-19, 1.867e-18, 5.974e-18, 
        1.366e-17, 2.554e-17, 6.222e-17, 1.136e-16, 1.747e-16, 
        2.740e-16, 4.330e-16, 6.681e-16, 7.998e-16, 8.649e-16, 
    ],
    r"D(d,n)3He ($cm^{-3} s^{-1}$)": [
        4.482e-28, 2.004e-26, 2.168e-25, 1.169e-24, 4.200e-24, 
        1.162e-23, 2.681e-23, 9.933e-23, 3.319e-22, 8.284e-22, 
        1.713e-21, 3.110e-21, 7.905e-21, 1.602e-20, 4.447e-20,
        9.128e-20, 1.573e-19, 3.457e-19, 6.023e-19, 9.175e-19, 
        1.481e-18, 2.603e-18, 5.271e-18, 8.235e-18, 1.133e-17, 
    ]
}

#-----------------------------------------------------------------------------
# Plot BH_tableVIII with caclulated Fusion Reactivity (Bosch-Hale Fits) for comparison

Ti_vals = BH_tableVIII["Ti (keV)"] # Extract tabulated values from the dictionary

# Use only values present in both D(t,n)α and D(d,n)³He
tdn_vals = BH_tableVIII[r"D(t,n)\alpha Reaction Rate ($cm^{-3} s^{-1}$)"]
ddn_vals = BH_tableVIII[r"D(d,n)3He ($cm^{-3} s^{-1}$)"]

# Compute fit values at the same Ti points
tdn_fit_vals = [sigma_v(T, reactivity_fits["T(d,n)4He"]) for T in Ti_vals]
ddn_fit_vals = [sigma_v(T, reactivity_fits["D(d,n)3He"]) for T in Ti_vals]

# Plot comparison
plt.figure(figsize=(10, 6))

# D(t,n)α
plt.scatter(Ti_vals, tdn_vals, color="green", label="Tabulated D(t,n)α", marker='o')
plt.plot(Ti_vals, tdn_fit_vals, color="green", label="Fit D(t,n)α")

# D(d,n)3He
plt.scatter(Ti_vals, ddn_vals, color="blue", label="Tabulated D(d,n)³He", marker='o')
plt.plot(Ti_vals, ddn_fit_vals, color="blue", label="Fit D(d,n)³He")

plt.yscale("log")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel("Ion Temperature T (keV)", fontsize=12)
plt.ylabel("Fusion Reactivity ⟨σv⟩ (cm³/s)", fontsize=12)
plt.title("Validation of Fusion Reactivity Bosch-Hale with Table VIII", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

