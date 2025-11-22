"""Neutron Spectra Calculation
1. take the calculated v_rel from each v1 and v2 pair in the distribution 
and calculate the reactivity as before but this time store each individual reactivity value
for each (v1, v2) pair into one larger array R_n.

2. use the equation to to calculate the energy distribution of the neutron E_n

E_n = 0.5 * m_n * (v_CM)**2 + (m_R/(m_n + m_R)) * (Q + K) + (v_CM) * cos(theta_CM) * sqrt(((2 * m_n * m_R)/(m_n + m_R)) * (Q +K))

where m_n = neutron mass
m_R = mass of residual particle in this case alpha (3He for DD and 4He for DT)
mu = reduced mass of incident particles
v_CM = (m1 * v1 + m2 * v2)/(m1 + m2) = velocity of the centre of mass
Q = released fusion energy in the reaction
K = total kinetic energy of incident particles in the centre of mass frame which can be found on the internet (= 0.5 * mu * (v_rel)**2)
theta_CM = the angle between neturon emission direction in the centre of mass frame and the centre of mass velocity

3. use the adjusted same equation to calulate the energy distribution of the alpha E_alpha
D + T = n + 4He
D + D = n + 3He

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/victo/OneDrive/Documents/SEPNet Internship UKAEA 2025/")
import BH_cross_section as BH_cs

# Physical constants
amu = 1.66053906660e-27     # atomic mass unit (kg)
m_n = 1.00866491588 * amu   # neutron mass (kg)
kB = 1.380649e-23           # Boltzmann constant (J/K)
eV_to_J = 1.602176634e-19
MeV_to_J = 1e6 * eV_to_J
J_to_keV = 6.2415e18 / 1e3  # J → keV
barn_to_m2 = 1e-28          # barn → m^2

# Q-values (MeV)
Q_values_MeV = {
    "DT": 17.589,
    "DD": 3.268
}

# Reaction definitions: (m1, m2, BH params, label)
reactions = {
    "DT": [3.3443e-27, 5.0082e-27, BH_cs.params_DT, "T(d,n)4He"], 
    "DD": [3.3443e-27, 3.3443e-27, BH_cs.params_DD, "D(d,n)3He"]
}

# Velocity grid 
v = np.linspace(0.0, 5e4, num=1000)         # ideally num = 5000
dv = np.insert(np.diff(v), -1, np.diff(v)[-1])

# Angle sampling (small!)
cos_theta_CM = np.linspace(-1, 1, num=5)   # test if num = 11 is ideal
dcos = np.abs(cos_theta_CM[1] - cos_theta_CM[0])

cos_rel = np.linspace(-1, 1, num=5)         # test if num = 11 is ideal
dcos_rel = np.abs(cos_rel[1] - cos_rel[0])

# Energy bins (also reduced for speed)
E_bins = {
    "DT": np.linspace(14-1, 14+1, num=51),   # 31 bins instead of 51
    "DD": np.linspace(2.45-0.5, 2.45+0.5, num=51)
}

# Temperature list
TI_keV_list = [5, 10, 15, 20]


# # Residual particle masses
# m_residual = {
#     "DT": 4.00260325413 * amu,   # 4He
#     "DD": 3.01602932265 * amu    # 3He
# }

# # Velocity grid
# v = np.linspace(0.0, 1e5, num=int(5e3))        # m/s 
# dv = np.insert(np.diff(v), -1, np.diff(v)[-1])  # same length as v

# # Angle sampling
# cos_theta_CM = np.linspace(-1, 1, num=11)  
# dcos = np.abs(cos_theta_CM[1] - cos_theta_CM[0]) 
# cos_rel = np.linspace(-1, 1, num=11)  
# dcos_rel = np.abs(cos_rel[1] - cos_rel[0])

# # Energy bins (MeV)
# E_bins = {
#     "DT": np.linspace(14-1, 14+1, num=51),
#     "DD": np.linspace(2.45-0.5, 2.45+0.5, num=51)
#}

def compute_neutron_spectra_fast(reaction, params, Ti_keV, v, dv,
                                 cos_rel, dcos_rel, cos_theta_CM, dcos):

    print(f"\nComputing reactivity for {reaction} @ {Ti_keV:.1f} keV...")
    start_time = time.time()

    # CHECKPOINT SETUP
    checkpoint_labels = [
        "Maxwellians computed",
        "Relative velocities computed",
        "Cross sections + kernels computed",
        "Neutron energies computed",
        "Packaging results"
    ]
    n_steps = len(checkpoint_labels)
    def checkpoint(i):
        elapsed = time.time() - start_time
        print(f"{reaction}: {int((i+1)/n_steps*100)}% — {checkpoint_labels[i]} "
              f"({elapsed:.2f} s, {elapsed/60:.2f} min)")

    # 1) Maxwellians & probability weights
    m1, m2, fit_params, _ = params
    mu = (m1 * m2) / (m1 + m2)

    Ti_eV = Ti_keV * 1e3

    pref1 = np.sqrt(2.0/np.pi) * (m1 / (kB * Ti_eV))**1.5
    pref2 = np.sqrt(2.0/np.pi) * (m2 / (kB * Ti_eV))**1.5
    MD1 = pref1 * v**2 * np.exp(-m1*v**2 / (2*kB*Ti_eV))
    MD2 = pref2 * v**2 * np.exp(-m2*v**2 / (2*kB*Ti_eV))
    MD1 /= np.trapz(MD1, v)
    MD2 /= np.trapz(MD2, v)

    MD_matrix = np.outer(MD1*dv, MD2*dv)   # (Nv,Nv)

    checkpoint(0)

    # 2) Compute relative velocities
    v1 = v[:, None, None]                 # (Nv,1,1)
    v2 = v[None, :, None]                 # (1,Nv,1)
    cos_rel3 = cos_rel[None, None, :]     # (1,1,Nθ_rel)

    v_rel = np.sqrt(np.maximum(0, v1**2 + v2**2 - 2*v1*v2*cos_rel3))

    checkpoint(1)

    # 3) COM motion, cross-section, kernel
    v_CM = (m1*v[:, None] + m2*v[None, :])/(m1+m2)

    K = 0.5 * mu * v_rel**2
    E_com_keV = K * J_to_keV

    sigma = BH_cs.sigma_bosch_hale(E_com_keV,
                                   fit_params['A'],
                                   fit_params['B'],
                                   fit_params['BG'])
    sigma = np.asarray(sigma)

    kernel = sigma * v_rel * barn_to_m2

    weight_rel = MD_matrix[:, :, None] * kernel * 0.5 * dcos_rel

    checkpoint(2)

    # 4) Neutron energy distribution
    m_residual = {
    "DT": 4.00260325413 * amu,   # 4He
    "DD": 3.01602932265 * amu    # 3He
    }
    
    mR = m_residual[reaction]

    Q_J = Q_values_MeV[reaction] * MeV_to_J

    sqrt_arg = ((2*m_n*mR)/(m_n+mR)) * (Q_J + K)
    S = np.sqrt(np.maximum(sqrt_arg, 0))

    A = 0.5*m_n*(v_CM[:, :, None]**2) + (mR/(m_n+mR))*(Q_J + K)

    cosCM = cos_theta_CM[None, None, None, :]  # (1,1,1,Nθ_CM)

    v_CM3 = v_CM[:, :, None]        # (Nv, Nv, 1)

    E_n = (A[..., None] + v_CM3[..., None] * cosCM * S[..., None])

    E_n_MeV = E_n / MeV_to_J

    # FIX WEIGHTS DIMENSIONS TO MATCH E_n
    Ncm = len(cos_theta_CM)

    weights = weight_rel[..., None] * (0.5 * dcos)
    weights = np.repeat(weights, Ncm, axis=3)


    checkpoint(3)
    result = {
        "E_n": E_n_MeV.ravel(),
        "R_n": weights.ravel()
    } 
    checkpoint(4)

    return result


# Compute histograms

neutron_spectra = {}

for reaction, params in reactions.items():
    for Ti_keV in TI_keV_list:
        result = compute_neutron_spectra_fast(reaction, params, Ti_keV, v, dv,
                                         cos_rel, dcos_rel, cos_theta_CM, dcos)

        neutron_spectra[(reaction, Ti_keV, "E_n")] = result["E_n"]
        neutron_spectra[(reaction, Ti_keV, "R_n")] = result["R_n"]

for reaction, params in reactions.items():
    for Ti_keV in TI_keV_list:
        E_n = neutron_spectra[(reaction, Ti_keV, "E_n")]
        R_n = neutron_spectra[(reaction, Ti_keV, "R_n")]
        hist, bins = np.histogram(E_n, bins=E_bins[reaction], weights=R_n)

        bin_centers = 0.5*(bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]
        spectrum_per_MeV = hist / bin_width

        neutron_spectra[(reaction, Ti_keV)] = {
            "E_bin_centers_MeV": bin_centers,
            "spectrum_per_MeV": spectrum_per_MeV,
        }

        print(f"Computed neutron spectrum for {reaction} @ {Ti_keV:.1f} keV")


# Plot histograms
for reaction in ["DT", "DD"]:
    plt.figure(figsize=(8,5))
    for Ti_keV in TI_keV_list:
        key = (reaction, Ti_keV)
        data = neutron_spectra[key]
        plt.step(data["E_bin_centers_MeV"], data["spectrum_per_MeV"],
                 where="mid", label=f"{Ti_keV} keV")
    plt.xlabel("Neutron Energy (MeV)")
    plt.ylabel("dR/dE (reactions / s / m³ / MeV)")
    plt.title(f"Neutron spectrum: {reaction}")
 #   plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
