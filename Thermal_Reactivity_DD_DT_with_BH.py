# Combined DT + DD Reactivity

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.interpolate import interp1d
import statsmodels.api as sm

sys.path.append("C:/Users/victo/OneDrive/Documents/SEPNet Internship UKAEA 2025/")
import BH_cross_section as BH_cs
import Bosch_Hale_Reactivity_Fit as BH_RF

# Constants
kB = 1.380649e-23  # J/K
J_to_keV = 6.2415e18 / 1e3     # J to keV

reactions = {
    "DT": [3.3443e-27, 5.0082e-27, BH_cs.params_DT, "T(d,n)4He"], # mass in kg
    "DD": [3.3443e-27, 3.3443e-27, BH_cs.params_DD, "D(d,n)3He"]
}

# Temperature and velocity grid
TI = np.linspace(0.5e3, 50e3, num=200)   # in eV num=200
v = np.arange(0.0, 5e6, 100000)         # m/s 10000 for higher resolution
dv = np.insert(np.diff(v), -1, np.diff(v)[-1])  # same length as v
xi = np.linspace(-1, 1, num=101)  # angular grid for cosθ

# Storage for results
reactivity = {}

for reaction, params in reactions.items():
    m1, m2, fit_params, BH_name = params
    reactivity[reaction] = np.zeros(len(TI))

    print(f"\nComputing reactivity for {reaction}...")

    start_time = time.time()
    n = len(v)
    total_iterations = len(TI) * n * n
    progress_check = int(0.05 * total_iterations)  # 5% steps
    iteration_count = 0
    next_checkpoint = progress_check

    for i in range(len(TI)):
        # Maxwellian distributions
        MD1 = np.sqrt(2/np.pi) * (m1 / (kB * TI[i] * 11604.5250061598))**(3/2) \
              * v**2 * np.exp(-m1 * v**2 / (2 * kB * TI[i] * 11604.5250061598))
        int_MD1 = np.trapz(MD1, v)
        MD1_norm = MD1 / int_MD1

        MD2 = np.sqrt(2/np.pi) * (m2 / (kB * TI[i] * 11604.5250061598))**(3/2) \
              * v**2 * np.exp(-m2 * v**2 / (2 * kB * TI[i] * 11604.5250061598))
        int_MD2 = np.trapz(MD2, v)
        MD2_norm = MD2 / int_MD2

        mu = (m1 * m2) / (m1 + m2)  # reduced mass

        for j in range(len(v)):
            v1 = v[j]
            for k in range(len(v)):
                v2 = v[k]
                v_rel = np.sqrt(v1**2 + v2**2 - 2.0 * v1 * v2 * xi)
                E_com_keV = 0.5 * mu * v_rel**2 * J_to_keV

                # Bosch–Hale cross-section
                sigma = BH_cs.sigma_bosch_hale(
                    E_com_keV,
                    fit_params['A'],
                    fit_params['B'],
                    fit_params['BG']
                )

                # Angular average
                angle_avg = 0.5 * np.trapz(v_rel * sigma, xi)

                # Contribution to ⟨σv⟩
                integral = MD1_norm[j] * dv[j] * MD2_norm[k] * dv[k] * angle_avg * 1e-28
                if not np.isnan(integral):
                    reactivity[reaction][i] += integral

            iteration_count += len(v)
            if iteration_count >= next_checkpoint:
                percent_done = (iteration_count / total_iterations) * 100
                elapsed = time.time() - start_time
                elapsed_minutes = elapsed / 60
                print(f"{reaction}: {percent_done:.0f}% done — {elapsed_minutes:.2f} minutes = {elapsed:.2f} s")
                next_checkpoint += progress_check
            

#-----------------------------------------------------------------------------------------

# Thermal Reactivity for DT and DD
plt.figure(figsize=(10,6))
for reaction, params in reactions.items():
    BH_name = params[3]

    plt.plot(TI/1e3, reactivity[reaction], label=f"{reaction} (SPARK)")

    # Bosch–Hale fit
    T_min, T_max = BH_RF.reactivity_fits[BH_name]["Ti_range_keV"]
    T_vals = np.linspace(T_min, T_max, 500)
    sv_vals = [BH_RF.sigma_v(T, BH_RF.reactivity_fits[BH_name]) for T in T_vals]
    plt.plot(T_vals, np.array(sv_vals)/1e6, linestyle="dashed", label=f"{reaction} (BH Fit)")

    # Bosch–Hale Table VIII values
    Ti_tab = BH_RF.BH_tableVIII["Ti (keV)"]
    if reaction == "DT":
        sv_tab = BH_RF.BH_tableVIII[r"D(t,n)\alpha Reaction Rate ($cm^{-3} s^{-1}$)"]
    else:
        sv_tab = BH_RF.BH_tableVIII[r"D(d,n)3He ($cm^{-3} s^{-1}$)"]

#    plt.scatter(Ti_tab, np.array(sv_tab)*1e-6, marker="o", label=f"{reaction} (BH Table VIII)")

plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.xlim(0, 50)
plt.xlabel("Ion Temperature T (keV)", fontsize=12)
plt.ylabel("Fusion Reactivity ⟨σv⟩ (m³/s)", fontsize=12)
plt.title("Comparison of Thermal Reactivities (DT vs DD)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------
# Relative Difference: SPARK vs BH Fit
plt.figure(figsize=(10,6))

for reaction, params in reactions.items():
    BH_name = params[3]

    # Bosch–Hale fit range + values
    T_min, T_max = BH_RF.reactivity_fits[BH_name]["Ti_range_keV"]
    T_vals = np.linspace(T_min, T_max, 500)
    sv_vals = [BH_RF.sigma_v(T, BH_RF.reactivity_fits[BH_name]) for T in T_vals]
    sv_vals = np.array(sv_vals) / 1e6  # Convert to m**3/s

    # Interpolation to match TI grid (keV)
    BH_interp = interp1d(T_vals, sv_vals, kind="linear", fill_value="extrapolate")
    BH_on_TI = BH_interp(TI/1e3)  # Convert TI from eV to keV

    # Relative difference (%)
    rel_diff = ((reactivity[reaction] - BH_on_TI) / BH_on_TI) * 100

    plt.plot(TI/1e3, rel_diff, label=f"{reaction} (SPARK vs BH Fit)")

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlim(0, 50)
plt.ylim(-35, 5)
plt.xlabel("Ion Temperature T (keV)", fontsize=12)
plt.ylabel("Relative Difference (%)", fontsize=12)
plt.title("Relative Difference Plot: SPARK vs Bosch–Hale Fit", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------

# Format the above "Thermal Reactivity for DT and DD" and "Relative Difference: SPARK vs BH Fit" so that both are aligned in the x axis and can be more easily compared.
# Add the relative difference graph underneath and fit the title so that there are no overlaps visually
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'hspace': 0.0}  # <-- no vertical gap
)

for reaction, params in reactions.items():
    BH_name = params[3]

    # SPARK-computed reactivity
    ax1.plot(TI/1e3, reactivity[reaction], label=f"{reaction} (SPARK)")

    # Bosch–Hale fit curve
    T_min, T_max = BH_RF.reactivity_fits[BH_name]["Ti_range_keV"]
    T_vals = np.linspace(T_min, T_max, 500)
    sv_vals = [BH_RF.sigma_v(T, BH_RF.reactivity_fits[BH_name]) for T in T_vals]
    ax1.plot(T_vals, np.array(sv_vals)/1e6, linestyle="dashed", label=f"{reaction} (BH Fit)")

ax1.set_yscale("log")
ax1.set_xlim(0, 50)
ax1.set_ylabel("Fusion Reactivity ⟨σv⟩ (m³/s)", fontsize=12)
ax1.set_title("Comparison of Thermal Reactivities (DT vs DD)", fontsize=14, pad=10)
ax1.grid(True, which="both", linestyle="--", alpha=0.6)
ax1.legend(loc="lower right")

# Reduce tick label overlap
plt.setp(ax1.get_xticklabels(), visible=False)

for reaction, params in reactions.items():
    BH_name = params[3]

    # Bosch–Hale fit interpolation
    T_min, T_max = BH_RF.reactivity_fits[BH_name]["Ti_range_keV"]
    T_vals = np.linspace(T_min, T_max, 500)
    sv_vals = [BH_RF.sigma_v(T, BH_RF.reactivity_fits[BH_name]) for T in T_vals]
    sv_vals = np.array(sv_vals) / 1e6

    BH_interp = interp1d(T_vals, sv_vals, kind="linear", fill_value="extrapolate")
    BH_on_TI = BH_interp(TI/1e3)

    # Relative difference (%)
    rel_diff = ((reactivity[reaction] - BH_on_TI) / BH_on_TI) * 100
    ax2.plot(TI/1e3, rel_diff, label=f"{reaction} (SPARK vs BH Fit)")

ax2.axhline(0, color="black", linestyle="--", linewidth=1)
ax2.set_xlim(0, 50)
ax2.set_ylim(-35, 0)
ax2.set_xlabel("Ion Temperature T (keV)", fontsize=12)
ax2.set_ylabel("Relative Difference (%)", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="lower right")

# Compact layout — no white space between subplots, but padding for titles
plt.subplots_adjust(hspace=0.1)
plt.show()

#COMPLETE and I am happy with the generated results!

#========================================================================================================================================
#========================================================================================================================================

# Reaction rate calculation using TRANSP dataset densities

import netCDF4  # Library for reading .cdf (NetCDF) files
from scipy.interpolate import interp1d

# Load NBI distribution data
FBMDat = netCDF4.Dataset(
    'C:/Users/victo/OneDrive/Documents/SEPNet Internship UKAEA 2025/NBI-distribution-function.cdf', 'r')

# Extract variables from the file
TimeFBM = np.array(FBMDat.variables['TIME']) # Time of NBI output

# Load background plasma data
PlasmaDat = netCDF4.Dataset(
    'C:/Users/victo/OneDrive/Documents/SEPNet Internship UKAEA 2025/Background-plasma.cdf', 'r')

# Read the time array from the plasma data
TimeArray = np.array(PlasmaDat.variables['TIME3'])

# Get the actual time value from TimeFBM
# If it's a scalar, extract its value; if it's an array, take the first entry
TimeFBM_val = TimeFBM.item() if np.ndim(TimeFBM) == 0 else TimeFBM[0]

# Find the index of the closest matching time
TimeIdx = np.argmin(np.abs(TimeArray - TimeFBM_val))

# Match the closest radial position in the plasma X-grid to the NBI radius
XArray = np.array(PlasmaDat.variables['X'])[TimeIdx,:]  # Shape: (radial variable at particular time)

# Extract plasma parameters at the chosen time and radius
TI = np.array(PlasmaDat.variables['TI'])[TimeIdx,:]  # Ion temperature [eV]
ND = np.array(PlasmaDat.variables['ND'])[TimeIdx, :] *1e6 # Deuterium density given in [cm^-3] originally by TRANSP
NT = np.array(PlasmaDat.variables['NT'])[TimeIdx,:]  *1e6 # Tritium density so we multiply by 1e6 her to give unitd of [m^-3]
NEUT_DT = np.array(PlasmaDat.variables['THNTX_DT'])[TimeIdx,:]  # Thermal neutron reaction rate DT
NEUT_DD = np.array(PlasmaDat.variables['THNTX_DD'])[TimeIdx,:]  # Thermal neutron reaction rate DD

# Maps reactant symbols to variable names in .cdf file
species_density_map = {
    "D": "ND",
    "d": "ND",
    "T": "NT",
    "t": "NT",
    "3He": "NHE3",
}

def parse_reactants(reaction_str):
    """Extract reactants from reaction string like 'D(d,n)3He' or 'T(d,n)4He'."""
    left = reaction_str.split('(')[0]   # e.g. 'D'
    inner = reaction_str.split('(')[1].split(')')[0]  # e.g. 'd,n'
    reactant1 = left.strip()
    reactant2 = inner.split(',')[0].strip()

    # Normalise cases
    normalization = {"d": "D", "t": "T", "h": "H"}
    reactant1 = normalization.get(reactant1, reactant1)
    reactant2 = normalization.get(reactant2, reactant2)

    return reactant1, reactant2

rho = np.array(PlasmaDat.variables['X'])[TimeIdx, :]  # radial coordinate (ρ)

reaction_rates = {}
for reaction, params in reactions.items():
    _, _, _, reaction_str = params
    n1_species, n2_species = parse_reactants(reaction_str)
    
    # reactivity[reaction] has units of m^3/s
    # but ND and NT from TRANSP is in cm^3/s so multiply by 1e6
    # n1, n2 are in m^-3
    # Reaction rate should be in [reactions/s/m^3].

    # Get density profiles
    n1_name = species_density_map[n1_species]
    n2_name = species_density_map[n2_species]
    n1 = np.array(PlasmaDat.variables[n1_name])[TimeIdx, :] * 1e6
    n2 = np.array(PlasmaDat.variables[n2_name])[TimeIdx, :] * 1e6
    TI_profile = np.array(PlasmaDat.variables['TI'])[TimeIdx, :]  # ion temp [eV], shape (nr,)

    # Interpolate reactivity (TI_grid to TI_profile)
    TI_grid_eV = np.linspace(0.5e3, 50e3, num=len(reactivity[reaction]))
    reactivity_interp = interp1d(TI_grid_eV, reactivity[reaction],
                                 kind="linear", fill_value="extrapolate")
    reactivity_on_TI = reactivity_interp(TI_profile)

    # Kronecker delta factor
    delta = 1 if n1_species == n2_species else 0

    # Direct integration reaction rate profile
    rate = (1 / (1 + delta)) * n1 * n2 * reactivity_on_TI
    reaction_rates[reaction] = rate

# Convert TRANSP rates to SI units
NEUT_DT_SI = NEUT_DT * 1e6  # cm^-3 -> m^-3
NEUT_DD_SI = NEUT_DD * 1e6

# Plot comparison --------------------
plt.figure(figsize=(8,6))

# Direct integration results
plt.plot(rho, reaction_rates["DT"], label="SPARK DT", linewidth=2)
plt.plot(rho, reaction_rates["DD"], label="SPARK DD", linewidth=2)

# TRANSP data (already a reaction rate profile vs rho)
plt.plot(rho, NEUT_DT_SI, '--', label="TRANSP THNTX_DT", color='C3', linewidth=1.5)
plt.plot(rho, NEUT_DD_SI, '--', label="TRANSP THNTX_DD", color='C4', linewidth=1.5)

plt.xlabel(r"$\rho_{tor}$")
plt.ylabel(r"$Reaction Rate (reactions/s/m^{3})$")
plt.yscale("log")
plt.title("Comparison of Reaction Rates: SPARK vs TRANSP")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------

# Relative Difference Graph for Reaction rate calculation using TRANSP dataset densities of SPARK vs TRANSP

plt.figure(figsize=(8,6))
EPS = 1e-30
mask_DT = np.abs(NEUT_DT_SI) > EPS
mask_DD = np.abs(NEUT_DD_SI) > EPS

rel_DT = np.zeros_like(NEUT_DT_SI)
rel_DD = np.zeros_like(NEUT_DD_SI)
rel_DT[mask_DT] = (reaction_rates["DT"][mask_DT] - NEUT_DT_SI[mask_DT]) / NEUT_DT_SI[mask_DT] * 100.0
rel_DD[mask_DD] = (reaction_rates["DD"][mask_DD] - NEUT_DD_SI[mask_DD]) / NEUT_DD_SI[mask_DD] * 100.0

plt.plot(rho, rel_DT, label="DT: SPARK vs TRANSP")
plt.plot(rho, rel_DD, label="DD: SPARK vs TRANSP")
plt.axhline(0, linestyle='--', linewidth=1, color='k')
plt.xlabel(r"$\rho_{tor}$")   # <-- fixed here
plt.ylabel("Relative Difference (%)")
plt.title("Relative Difference: Reaction Rate Profiles SPARK vs TRANSP")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------
# Combined graph for ease of comparison

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 9), sharex=True, gridspec_kw={'hspace': 0.0}  # no gap between subplots
)

ax1.plot(rho, reaction_rates["DT"], label="SPARK DT", linewidth=2)
ax1.plot(rho, reaction_rates["DD"], label="SPARK DD", linewidth=2)

# TRANSP data (converted to SI units)
ax1.plot(rho, NEUT_DT_SI, '--', label="TRANSP THNTX_DT", color='C3', linewidth=1.5)
ax1.plot(rho, NEUT_DD_SI, '--', label="TRANSP THNTX_DD", color='C4', linewidth=1.5)

ax1.set_yscale("log")
ax1.set_ylabel(r"Reaction Rate (reactions/s/m$^{3}$)", fontsize=12)
ax1.set_title("Comparison of Reaction Rates: SPARK vs TRANSP", fontsize=14, pad=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc="lower left")
plt.setp(ax1.get_xticklabels(), visible=False)  # hide overlapping x labels

EPS = 1e-30
mask_DT = np.abs(NEUT_DT_SI) > EPS
mask_DD = np.abs(NEUT_DD_SI) > EPS

rel_DT = np.zeros_like(NEUT_DT_SI)
rel_DD = np.zeros_like(NEUT_DD_SI)
rel_DT[mask_DT] = (reaction_rates["DT"][mask_DT] - NEUT_DT_SI[mask_DT]) / NEUT_DT_SI[mask_DT] * 100.0
rel_DD[mask_DD] = (reaction_rates["DD"][mask_DD] - NEUT_DD_SI[mask_DD]) / NEUT_DD_SI[mask_DD] * 100.0

ax2.plot(rho, rel_DT, label="DT: SPARK vs TRANSP", linewidth=2)
ax2.plot(rho, rel_DD, label="DD: SPARK vs TRANSP", linewidth=2)
ax2.axhline(0, linestyle='--', linewidth=1, color='k')

ax2.set_xlabel(r"$\rho_{tor}$", fontsize=12)
ax2.set_ylabel("Relative Difference (%)", fontsize=12)
#ax2.set_title("Relative Difference: Reaction Rate Profiles SPARK vs TRANSP", fontsize=14, pad=8)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc="lower left")

plt.subplots_adjust(hspace=0.0, top=0.95, bottom=0.08)
plt.show()
