import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# EQUATIONS

def S_bosch_hale(E, A, B):
    """
    Compute S(E) for given energy E using Bosch–Hale fit:
    S(E) = (A1 + E(A2 + E(A3 + E(A4 + EA5)))) / (1 + E(B1 + E(B2 + E(B3 + EB4))))
    """
    numerator = A[0] + E * (A[1] + E * (A[2] + E * (A[3] + E * A[4])))
    denominator = 1 + E * (B[0] + E * (B[1] + E * (B[2] + E * B[3])))
    return numerator / denominator  # in keV·mbarn

def sigma_bosch_hale(E, A, B, BG):
    """
    Compute cross section σ(E) in barns from S(E) using:
    σ(E) = [S(E)/E] * exp(-BG / sqrt(E))
    """
    S_E = S_bosch_hale(E, A, B)
    sigma = (S_E / E) * np.exp(-BG / np.sqrt(E))
    return sigma / 1000  # Convert from mbarn to barn


# -----------------------------------------------
# PARAMETERS

#Bosch-Hale
# D(t,n)alpha reaction
params_DT = {
    'name': 'D(t,n)α',
    'BG': 34.3827,
    'A': [6.927E+4, 7.454E+8, 2.05E+6, 5.2002E+4, 0.0],
    'B': [6.38E+1, -9.95E-1, 6.981E-5, 1.728E-4],
    'E_range': (0.5, 550),
    'deltaS_max': 1.9  # %
}

# D(d,n)3He reaction
params_Dt = {
    'name': 'D(t,n)³He at high energies',
    'BG': 34.3827,
    'A': [-1.4714E+6, 0.0, 0.0, 0.0, 0.0],
    'B': [-8.4127E-3, 4.7983E-6, -1.0748E-9, 8.5184E-14],
    'E_range': (550, 4700),
    'deltaS_max': 2.5  # %
}

# D(d,n)3He reaction
params_DD = {
    'name': 'D(d,n)³He',
    'BG': 31.3970,
    'A': [5.3701E+4, 3.3027E+2, -1.2706E-1, 2.9327E-5, -2.511E-9],
    'B': [0.0, 0.0, 0.0, 0.0],
    'E_range': (0.5, 4900),
    'deltaS_max': 2.5  # %
}

# Tentori 2023
params_tentori_2023 = {
    'name': 'Tentori 2023',
    'E_range': (0, 9.76E+3), # Energy range in keV

    # Region 1: E ≤ 400 keV
    'region_1': {
        'range': (0, 4.00E+2),
        'C': [1.97E+5, 2.69E+2, 2.54E-1],  # [C0, C1, C2] in keV·b
        'resonance': {
            'AL': 0.00E+0,                # keV·b
            'EL': 0.00E+0,                # keV
            'delta_EL': 0.00E+0           # keV
        }
    },

    # Region 2: 400 < E ≤ 668 keV
    'region_2': {
        'range': (4.00E+2, 6.68E+2),
        'D': [3.46E+5, 1.50E+5, -5.99E+4, -4.60E+2],  # [D0, D1, D2, D5] in keV·b
        'E0': 4.00E+2  # base energy offset (ε₁)
    },

    # Region 3: 668 < E ≤ 9760 keV
    'region_3': {
        'range': (6.68E+2, 9.76E+3),
        'B': 3.81E+2,  # keV·b
        'A': [1.98E+9, 3.89E+9, 1.36E+9, 3.71E+9],  # A₀–A₃ in keV·b
        'E': [6.409E+2, 1.211E+3, 2.340E+3, 3.294E+3],  # E₀–E₃ in keV
        'delta_E': [8.55E+1, 4.14E+2, 2.21E+2, 3.51E+2]  # δE₀–δE₃ in keV
    }
}



# -----------------------------------------------
# Plot Cross Sections

def plot_cross_sections(params_list):
    plt.figure(figsize=(9, 6))
    for params in params_list:
        E = np.linspace(params['E_range'][0], params['E_range'][1], 100000)
        sigma = sigma_bosch_hale(E, params['A'], params['B'], params['BG'])
        plt.plot(E, sigma, label=f"{params['name']} (ΔSmax = {params['deltaS_max']}%)")

    plt.xlabel("Energy in Centre-of-Mass Frame [keV]")
    plt.ylabel("Cross Section [barn]")
    plt.title("Fusion Cross Sections via Bosch–Hale Parameterisation")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the plot
plot_cross_sections([params_DT, params_DD, params_Dt,])
