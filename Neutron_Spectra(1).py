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

# Velocity grid \
v = np.linspace(0.0, 5e5, num=500)         # ideally num = 5000
dv = np.insert(np.diff(v), -1, np.diff(v)[-1]) # same length as v

# Angle sampling
cos_theta_CM = np.linspace(-1, 1, num=11)   # test if num = 11 is ideal
dcos = np.abs(cos_theta_CM[1] - cos_theta_CM[0])

cos_rel = np.linspace(-1, 1, num=11)       
dcos_rel = np.abs(cos_rel[1] - cos_rel[0])

# Energy bins
E_bins = {
    "DT": np.linspace(14-1, 14+1, num=401),   # 51-201 for testing
    "DD": np.linspace(2.45-0.5, 2.45+0.5, num=401)
}

# Temperature list
TI_keV_list = [5, 10, 15, 20]


def compute_neutron_spectra_fast(reaction, params, Ti_keV, v, dv,
                                 cos_rel, dcos_rel, cos_theta_CM, dcos):
    """
    Memory-efficient computation of neutron energy samples and weights.

    Returns:
        {"E_n": E_n_MeV_flat, "R_n": weights_flat}
    where both arrays are 1-D and have the same length: Nv*Nv*Ncos_rel*Ncm
    """
    print(f"\nComputing reactivity for {reaction} @ {Ti_keV:.1f} keV...")
    t0 = time.time()

    # Unpack masses/params
    m1, m2, fit_params, _ = params
    mu = (m1 * m2) / (m1 + m2)

    # Temperatures (eV)
    Ti_eV = Ti_keV * 1e3

    # Maxwellians (normalised)
    pref1 = np.sqrt(2.0/np.pi) * (m1 / (kB * Ti_eV))**1.5
    pref2 = np.sqrt(2.0/np.pi) * (m2 / (kB * Ti_eV))**1.5
    MD1 = pref1 * v**2 * np.exp(-m1*v**2 / (2*kB*Ti_eV))
    MD2 = pref2 * v**2 * np.exp(-m2*v**2 / (2*kB*Ti_eV))
    MD1 /= np.trapz(MD1, v)
    MD2 /= np.trapz(MD2, v)

    MD_matrix = np.outer(MD1 * dv, MD2 * dv)   # shape (Nv, Nv)
    Nv = v.size

    # Prepare velocity grids
    v1 = v[:, None, None]            # (Nv,1,1)
    v2 = v[None, :, None]            # (1,Nv,1)
    cos_rel3 = cos_rel[None, None, :]# (1,1,Ncos_rel)

    # Relative velocities and COM motion
    v_rel = np.sqrt(np.maximum(0.0, v1**2 + v2**2 - 2.0*v1*v2*cos_rel3))  # (Nv,Nv,Ncos_rel)
    v_CM = (m1 * v[:, None] + m2 * v[None, :]) / (m1 + m2)                 # (Nv,Nv)

    # Kinetic energy in COM (J)
    K = 0.5 * mu * v_rel**2                     # (Nv,Nv,Ncos_rel)
    E_com_keV = K * J_to_keV
    E_com_keV = np.maximum(E_com_keV, 1e-12)    # avoid E=0 issues in BH function

    # Cross-section (Bosch-Hale) — ensure no NaN/infs
    sigma = BH_cs.sigma_bosch_hale(E_com_keV,
                                   fit_params['A'],
                                   fit_params['B'],
                                   fit_params['BG'])
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # kernel and weight_rel (Nv,Nv,Ncos_rel)
    kernel = sigma * v_rel * barn_to_m2
    weight_rel = MD_matrix[:, :, None] * kernel * 0.5 * dcos_rel

    # Residual mass and Q (J)
    m_residual = {
        "DT": 4.00260325413 * amu,
        "DD": 3.01602932265 * amu
    }
    mR = m_residual[reaction]
    Q_J = Q_values_MeV[reaction] * MeV_to_J

    # Precompute A and S arrays with shape (Nv,Nv,Ncos_rel)
    # A = 0.5*m_n*v_CM^2 + (mR/(m_n+mR))*(Q + K)
    v_CM_sq = v_CM**2                              # (Nv,Nv)
    A_base = 0.5 * m_n * v_CM_sq[:, :, None]       # (Nv,Nv,1) broadcasts with K
    A = A_base + (mR / (m_n + mR)) * (Q_J + K)     # (Nv,Nv,Ncos_rel)

    sqrt_arg = ((2.0 * m_n * mR) / (m_n + mR)) * (Q_J + K)
    S = np.sqrt(np.maximum(sqrt_arg, 0.0))         # (Nv,Nv,Ncos_rel)

    # Now loop over cos(theta_CM) — keeps memory small
    Ncm = len(cos_theta_CM)
    E_blocks = []   # collect flattened E samples per CM angle
    W_blocks = []   # collect flattened weights per CM angle

    # precompute factor used for theta sampling (same for each cos theta)
    theta_weight_factor = 0.5 * dcos   # matches your previous implementation

    for cos_cm in cos_theta_CM:
        # E_n for this CM angle (Nv,Nv,Ncos_rel)
        # E_n = A + v_CM * cos_cm * S
        E_n_j = A + (v_CM[:, :, None] * cos_cm) * S    # (Nv,Nv,Ncos_rel)

        # weight for this CM angle (Nv,Nv,Ncos_rel)
        w_j = weight_rel * theta_weight_factor        # same shape (Nv,Nv,Ncos_rel)

        # Append flattened
        E_blocks.append((E_n_j / MeV_to_J).ravel())   # convert to MeV and flatten
        W_blocks.append(w_j.ravel())

    # Concatenate along CM angles to make final vectors
    E_n_MeV_flat = np.concatenate(E_blocks)
    R_n_flat = np.concatenate(W_blocks)

    # Sanity checks
    if E_n_MeV_flat.size != R_n_flat.size:
        raise RuntimeError("E_n and R_n sizes mismatch after concatenation")

    t_elapsed = time.time() - t0
    print(f"{reaction}: finished in {t_elapsed:.2f} s — samples: {E_n_MeV_flat.size}")

    return {"E_n": E_n_MeV_flat, "R_n": R_n_flat}

#-----------------------------------------------------------------------------------------------------
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


# Plot histograms seperately 
for reaction in ["DT", "DD"]:
    plt.figure(figsize=(8,5))
    for Ti_keV in TI_keV_list:
        key = (reaction, Ti_keV)
        data = neutron_spectra[key]
        plt.step(data["E_bin_centers_MeV"], data["spectrum_per_MeV"],
                 where="mid", label=f"{Ti_keV} keV")
    plt.xlabel("Neutron Energy (MeV)")
    plt.ylabel("dR/dE (reactions / s / m³ / MeV)")
    plt.title(f"Neutron Spectrum: {reaction}")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# Plot histograms together
fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

xlims = {
    "DT": (13.8, 14.25),
    "DD": (2.35, 2.5),
}

for ax, reaction in zip(axes, ["DD", "DT"]):
    for Ti_keV in TI_keV_list:
        key = (reaction, Ti_keV)
        data = neutron_spectra[key]

        ax.step(
            data["E_bin_centers_MeV"],
            data["spectrum_per_MeV"],
            where="mid",
            label=f"{Ti_keV} keV"
        )
    ax.set_xlim(xlims[reaction])
    ax.set_title(f"{reaction} Spectrum")
    ax.set_xlabel("Neutron Energy (MeV)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

axes[0].set_ylabel("dR/dE (reactions / s / m³ / MeV)")
axes[1].legend(title="Ion Temperature", fontsize=9)

plt.tight_layout()
plt.show()

