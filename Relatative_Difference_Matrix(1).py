import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.interpolate import interp1d

# UI for popup global progress bar
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except Exception:
    GUI_AVAILABLE = False

sys.path.append("C:/Users/victo/OneDrive/Documents/SEPNet Internship UKAEA 2025/")
import BH_cross_section as BH_cs
import Bosch_Hale_Reactivity_Fit as BH_RF

import os
import pickle

CHECKPOINT_FILE = "reactivity_checkpoint.pkl"

def save_checkpoint(reactivity, rel_diff_matrix):
    """Saves current progress to disk."""
    data = {
        "reactivity": reactivity,
        "rel_diff_matrix": rel_diff_matrix
    }
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(data, f)

def load_checkpoint():
    """Loads checkpoint if it exists, otherwise returns empty dicts."""
    if os.path.exists(CHECKPOINT_FILE):
        print("Loading previous checkpoint...")
        with open(CHECKPOINT_FILE, "rb") as f:
            data = pickle.load(f)
        return data["reactivity"], data["rel_diff_matrix"]
    else:
        print("No checkpoint found. Starting fresh.")
        return {}, {reaction: np.zeros((len(v_grid), len(TI_grid)))
                    for reaction in reactions}


kB = 1.380649e-23  # J/K
J_to_keV = 6.2415e18 / 1e3     # J to keV

reactions = {
    "DT": [3.3443e-27, 5.0082e-27, BH_cs.params_DT, "T(d,n)4He"], # mass in kg
    "DD": [3.3443e-27, 3.3443e-27, BH_cs.params_DD, "D(d,n)3He"]
}

# Temperature and velocity grid
TI_grid = [5, 10, 100, 1000]
v_grid = [10, 100, 1000, 5e3, 1e4] 

#TI_grid = [5, 10, 100, 1000, 10000] # resolution of TI = no of TI points = 5 columns
#v_grid = [10, 100, 1000, 5e3, 1e4, 5e4, 1e5, 1e6] # resolution of v = no of v points = 8 rows

# Storage for results
reactivity, rel_diff_matrix = load_checkpoint()

# Ensure matrices exist for all reactions
for reaction in reactions:
    if reaction not in rel_diff_matrix:
        rel_diff_matrix[reaction] = np.zeros((len(v_grid), len(TI_grid))) # 8x5matrix x2 for DD and DT

# -----------------------------------------------------------------------------------
# Global total iterations for the global progress bar
global_total_iterations = 0

for M in TI_grid:
    for N in v_grid:
        n = int(N)
        # For each reaction we run len(TI) * n * n inner iterations
        global_total_iterations += len(reactions) * int(M) * n * n

global_progress = 0  # increments will be in "iteration units" consistent with total

# Helper to format seconds prettily
def format_time(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m:d}m {s:02d}s"
    else:
        return f"{s:d}s"

# Initialise popup progress window
if GUI_AVAILABLE:
    try:
        root = tk.Tk()
        root.title("Total Progress")
        root.geometry("480x120")
        root.resizable(False, False)

        progress_var = tk.IntVar(value=0)

        ttk.Label(root, text="Overall progress:").pack(pady=(10, 0))
        progress_bar = ttk.Progressbar(root, orient="horizontal",
                                       length=440, mode="determinate",
                                       maximum=global_total_iterations,
                                       variable=progress_var)
        progress_bar.pack(pady=(6, 4))

        status_label = ttk.Label(root, text="Starting...")
        status_label.pack()

        # Force initial draw
        root.update_idletasks()
    except Exception:
        GUI_AVAILABLE = False

#-----------=============================------------------------=====================----------------
global_start_time = time.time()

# To avoid updating GUI too often, compute an update step
if global_total_iterations > 0:
    gui_update_step = max(1, global_total_iterations // 500)  # ~500 updates across whole run
else:
    gui_update_step = 1

# Main loops
for i_M, M in enumerate(TI_grid):     # i_M = index, M = actual value
    TI = np.linspace(0.5e3, 50e3, num=int(M))
    for i_N, N in enumerate(v_grid):  # i_N = index, N = actual value
        v = np.linspace(0.0, 5e6, num=int(N))  # cast N to int, units m/s
        dv = np.insert(np.diff(v), -1, np.diff(v)[-1])  # same length as v
        xi = np.linspace(-1, 1, num=101)  # angular grid for cosθ

        for reaction, params in reactions.items():
            m1, m2, fit_params, BH_name = params
    
            # Create storage if missing
            if reaction not in reactivity:
                reactivity[reaction] = np.zeros(len(TI))
    
            # --- SKIP COMPUTATION IF THIS (i_N, i_M) ALREADY FINISHED ---
            if rel_diff_matrix[reaction][i_N, i_M] != 0:
                print(f"Skipping: reaction={reaction}, TI={M}, v={N} (already computed)")
                continue

            print(f"\nComputing reactivity for {reaction} at TI = {M} eV and v = {N} m/s ...")

            start_time = time.time()
            n = len(v)
            total_iterations = len(TI) * n * n
            progress_check = int(0.05 * total_iterations)  # 5% steps
            if progress_check <= 0:
                progress_check = 1
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

                        # Contribution to reactivity
                        integral = MD1_norm[j] * dv[j] * MD2_norm[k] * dv[k] * angle_avg * 1e-28
                        if not np.isnan(integral):
                            reactivity[reaction][i] += integral
                        
                        #--------------------------------------------------------------------------------

                    # finished the inner k-loop for this j -> update local iteration counter
                    iteration_count += len(v)

                    # Update global progress
                    global_progress += len(v)

                    # Print periodic console updates for progress tracking
                    if iteration_count >= next_checkpoint:
                        percent_done = (iteration_count / total_iterations) * 100
                        elapsed = time.time() - start_time
                        elapsed_minutes = elapsed / 60
                        print(f"{reaction}: {percent_done:.0f}% done — {elapsed_minutes:.2f} minutes = {elapsed:.2f} s")
                        next_checkpoint += progress_check

                    # Update GUI progress tracking
                    if global_progress % gui_update_step == 0 or global_progress >= global_total_iterations:
                        elapsed_total = time.time() - global_start_time
                        pct_total = (global_progress / global_total_iterations) * 100 if global_total_iterations > 0 else 100.0
                        # Estimate remaining time
                        if global_progress > 0:
                            est_total = elapsed_total * (global_total_iterations / global_progress)
                            eta = est_total - elapsed_total
                            eta_str = format_time(eta)
                        else:
                            eta_str = "?"
                        status_text = f"{global_progress}/{global_total_iterations} ({pct_total:.2f}%) — elapsed {format_time(elapsed_total)} — ETA {eta_str}"

                        if GUI_AVAILABLE:
                            try:
                                progress_var.set(global_progress)
                                status_label.config(text=status_text)
                                # force the GUI to refresh
                                root.update_idletasks()
                                root.update()
                            except Exception:
                                # if GUI dies mid-run, fall back to console prints
                                GUI_AVAILABLE = False
                                print(status_text)
                        else:
                            print(status_text)

        # Relative percentage difference between computed reactivity and Bosch–Hale fit
        # Bosch–Hale fit range + values (BH expects keV)
        # Note: the original code referenced 'reaction' here; keep consistent with intent by computing for each reaction.
        for reaction, params in reactions.items():
            # compute rel_diff for this reaction and store in rel_diff_matrix at the [i_N, i_M] pos
            # BH expects keV; TI is eV
            T_min, T_max = BH_RF.reactivity_fits[params[3]]["Ti_range_keV"]
            T_vals = np.linspace(T_min, T_max, 500)
            sv_vals = [BH_RF.sigma_v(T, BH_RF.reactivity_fits[params[3]]) for T in T_vals]
            sv_vals = np.array(sv_vals) / 1e6  # Convert to m**3/s if BH returns cm^3/s or similar

            # Interpolate BH fit to your TI grid (convert TI eV -> keV)
            BH_interp = interp1d(T_vals, sv_vals, kind="linear", fill_value="extrapolate")
            BH_on_TI = BH_interp(TI / 1e3)  # TI in eV -> /1e3 to keV

            # Relative difference (%). Avoid division-by-zero by masking BH_on_TI == 0
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = ((reactivity[reaction] - BH_on_TI) / BH_on_TI) * 100.0
                rel_diff = np.where(np.isfinite(rel_diff), rel_diff, np.nan)  # replace inf/nan with nan

                mean_rel_diff = np.nanmean(rel_diff)
                save_checkpoint(reactivity, rel_diff_matrix) # SAVE CHECKPOINT after finishing this matrix entry
                print(f"Checkpoint saved for {reaction} at TI={M}, v={N}")


# End of main loops
total_elapsed = time.time() - global_start_time
print("\nAll computations finished.")
print(f"Total elapsed time: {format_time(total_elapsed)} ({total_elapsed:.2f} seconds)")

# Plot Graph --------------------------------------------------------------------------------------------------------
# convert v_grid to numeric array for plotting
x_vals = np.array([int(v) for v in v_grid])

# Prepare colors for each TI resolution
n_TI = len(TI_grid)
cmap = plt.get_cmap('viridis')
colors = [cmap(i / max(1, n_TI-1)) for i in range(n_TI)]

# Create figure: place subplots side-by-side (1 row, n_reac columns)
reactions_list = list(rel_diff_matrix.keys())
n_reac = len(reactions_list)
fig, axes = plt.subplots(nrows=1, ncols=n_reac, figsize=(4 * n_reac, 6), sharex=True)

# Ensure axes is always a 1D array we can iterate over
axes = np.atleast_1d(axes)

# Plot and gather handles for a single shared legend
all_handles = []
all_labels = []
for ax, reaction in zip(axes, reactions_list):
    mat = rel_diff_matrix[reaction]  # shape (len(v_grid), len(TI_grid))
    for ti_idx, TI_value in enumerate(TI_grid):
        y = mat[:, ti_idx]  # shape (len(v_grid),)
        mask = ~np.isnan(y)
        if np.any(mask):
            h = ax.scatter(x_vals[mask], y[mask],
                           label=f"TI = {TI_value}",
                           color=colors[ti_idx],
                           s=50, alpha=0.85, edgecolors='k', linewidths=0.3)
            # Save handles/labels once (avoid duplicates)
            if ti_idx == 0 and reaction == reactions_list[0]:
                # only from the first axis capture handles/labels in-order across TI
                pass
            # We only want one set of handles/labels for the figure legend, so capture from first axis:
            if reaction == reactions_list[0]:
                all_handles.append(h)
                all_labels.append(f"TI = {TI_value}")

    ax.set_xscale('log')   # helpful because v_grid spans orders of magnitude
    ax.set_ylabel("Relative Difference (%)", fontsize=12)
    ax.set_xlabel("Resolution of Velocity Grid (num of v points)", fontsize=12)
    ax.set_title(f"Relative % Difference - {reaction} Reaction", fontsize=13)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

# Create a single figure legend at the right of the axes so it doesn't cover points
# Adjust subplot area to make room on the right for the legend
plt.subplots_adjust(right=0.78)  # leave room on the right

# Place shared legend outside the axes (centered vertically)
fig.legend(all_handles, all_labels,
           title="TI (TI_grid)",
           loc='center right',
           bbox_to_anchor=(0.98, 0.5),   # slightly outside the figure on the right
           frameon=True,
           framealpha=0.95,
           borderpad=0.6)

# place legend below the plots 
plt.subplots_adjust(bottom=0.18)  # make room below for the legend
fig.legend(all_handles, all_labels, title="TI (num points)", loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=n_TI)

plt.tight_layout()
plt.show()


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
ND = np.array(PlasmaDat.variables['ND'])[TimeIdx, :]  # Deuterium density [cm^-3]
NT = np.array(PlasmaDat.variables['NT'])[TimeIdx,:]  # Tritium density [cm^-3]
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

# Plot comparison --------------------
plt.figure(figsize=(8,6))

# Direct integration results
plt.plot(rho, reaction_rates["DT"], label="Direct Integration DT", linewidth=2)
plt.plot(rho, reaction_rates["DD"], label="Direct Integration DD", linewidth=2)

# TRANSP data (already a reaction rate profile vs rho)
plt.plot(rho, NEUT_DT, '--', label="TRANSP THNTX_DT", color='C3', linewidth=1.5)
plt.plot(rho, NEUT_DD, '--', label="TRANSP THNTX_DD", color='C4', linewidth=1.5)

plt.xlabel(r"$\rho_{tor}$")
plt.ylabel("Reaction Rate (reactions/s/m^3)")
plt.yscale("log")
plt.title("Comparison of Reaction Rates: Direct Integration vs TRANSP")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
