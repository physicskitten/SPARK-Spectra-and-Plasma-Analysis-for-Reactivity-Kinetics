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

# Residual particle masses
m_residual = {
    "DT": 4.00260325413 * amu,   # 4He
    "DD": 3.01602932265 * amu    # 3He
}

# Velocity grid
v = np.linspace(0.0, 1e5, num=int(5e3))        # m/s 
dv = np.insert(np.diff(v), -1, np.diff(v)[-1])  # same length as v

# Angle sampling
cos_theta_CM = np.linspace(-1, 1, num=11)  
dcos = np.abs(cos_theta_CM[1] - cos_theta_CM[0]) 
cos_rel = np.linspace(-1, 1, num=11)  
dcos_rel = np.abs(cos_rel[1] - cos_rel[0])

# Energy bins (MeV)
E_bins = {
    "DT": np.linspace(14-1, 14+1, num=51),
    "DD": np.linspace(2.45-0.5, 2.45+0.5, num=51)
}

# Temperatures to run
TI_keV_list = [10]


def compute_neutron_spectra(reaction, params, Ti_keV, v, dv, cos_rel, dcos_rel, cos_theta_CM, dcos):
    """Compute raw neutron energies and weights for histogramming."""
    m1, m2, fit_params, _ = params
    mu = (m1 * m2) / (m1 + m2)

    Ti_eV = Ti_keV * 1e3

    # Maxwellian distributions
    pref1 = np.sqrt(2.0 / np.pi) * (m1 / (kB * Ti_eV))**1.5
    pref2 = np.sqrt(2.0 / np.pi) * (m2 / (kB * Ti_eV))**1.5
    MD1 = pref1 * v**2 * np.exp(-m1 * v**2 / (2.0 * kB * Ti_eV))
    MD2 = pref2 * v**2 * np.exp(-m2 * v**2 / (2.0 * kB * Ti_eV))
    MD1 /= np.trapz(MD1, v)
    MD2 /= np.trapz(MD2, v)

    # Storage
    E_n_all, R_n_all = [], []

    mR = m_residual[reaction]
    Q_J = Q_values_MeV[reaction] * MeV_to_J

    # Progress tracking
    nv = len(v)
    total_iterations = nv * nv
    progress_check = int(0.05 * total_iterations)  # print every 5%
    iteration_count = 0
    next_checkpoint = progress_check
    start_time = time.time()

    print(f"\nComputing reactivity for {reaction} @ {Ti_keV:.1f} keV...")

    for j, v1 in enumerate(v):
        MD1_j = MD1[j] * dv[j]
        for k, v2 in enumerate(v):
            MD2_k = MD2[k] * dv[k]

            v_rel_arr = np.sqrt(np.maximum(0.0, v1**2 + v2**2 - 2.0*v1*v2*cos_rel))
            v_CM = (m1*v1 + m2*v2)/(m1+m2)
            K_arr_J = 0.5 * mu * v_rel_arr**2
            E_com_keV_arr = K_arr_J * J_to_keV

            sigma_arr = BH_cs.sigma_bosch_hale(E_com_keV_arr,
                                               fit_params['A'],
                                               fit_params['B'],
                                               fit_params['BG'])
            sigma_arr = np.array(sigma_arr, copy=False)
            kernel_arr = v_rel_arr * sigma_arr * barn_to_m2

            for idx_rel in range(len(cos_rel)):
                k_val = kernel_arr[idx_rel]
                if k_val <= 0 or np.isnan(k_val):
                    continue

                K_J = K_arr_J[idx_rel]
                pref_w = MD1_j * MD2_k * k_val * 0.5 * dcos_rel

                sqrt_arg = ((2.0*m_n*mR)/(m_n+mR))*(Q_J+K_J)
                if sqrt_arg < 0:
                    continue
                S = np.sqrt(sqrt_arg)

                A = 0.5*m_n*v_CM**2 + (mR/(m_n+mR))*(Q_J+K_J)

                # Neutron energies (all emission angles)
                E_n_J = A + v_CM*cos_theta_CM*S
                E_n_all.append(E_n_J / MeV_to_J)

                # Reactivity weights
                weights = pref_w * 0.5 * dcos * np.ones_like(E_n_J)
                R_n_all.append(weights)


            # Progress update
            iteration_count += 1
            if iteration_count >= next_checkpoint:
                percent_done = (iteration_count / total_iterations) * 100
                elapsed = time.time() - start_time
                elapsed_minutes = elapsed / 60
                print(f"{reaction}: {percent_done:.0f}% done — {elapsed_minutes:.2f} minutes = {elapsed:.2f} s")
                next_checkpoint += progress_check

    return {"E_n": np.concatenate(E_n_all),
            "R_n": np.concatenate(R_n_all)}



# Compute histograms

neutron_spectra = {}

for reaction, params in reactions.items():
    for Ti_keV in TI_keV_list:
        result = compute_neutron_spectra(reaction, params, Ti_keV, v, dv,
                                         cos_rel, dcos_rel, cos_theta_CM, dcos)

        E_n, R_n = result["E_n"], result["R_n"]

#Post processing
for reaction, params in reactions.items():
    for Ti_keV in TI_keV_list:
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
