import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lmom import *
from scipy.stats import kappa4
import scipy.stats as sstats
from scipy.optimize import curve_fit
from scipy.special import inv_boxcox 
import math 
from scipy import stats

###############################################################################
# Section 1: Data Setup
###############################################################################

# Load data files
AnnualStatsNew = np.genfromtxt('Compiled_Annual_Stats_Newer.csv', delimiter=',', skip_header=1)
AnnualStatsOld = np.genfromtxt('Compiled_Annual_Stats_Older.csv', delimiter=',', skip_header=1)
CompiledStats = pd.read_excel('GaugingLocations.xlsx', index_col=None, header=0)

# Combine and organize annual maximum discharge data
D_AreaNew = AnnualStatsNew[:,1]
Ann_MaxNew = AnnualStatsNew[:,5]
D_AreaOld = AnnualStatsOld[:,1]
Ann_MaxOld = AnnualStatsOld[:,5]

D_Area = np.concatenate((D_AreaNew, D_AreaOld), axis=0)
Ann_Max = np.concatenate((Ann_MaxNew, Ann_MaxOld), axis=0)

# Sort data by drainage area
AreaVsMax = np.column_stack((D_Area, Ann_Max))
sorted_indices = np.argsort(AreaVsMax[:, 0])
AreaVsMax = AreaVsMax[sorted_indices]

# Group data by basin
changes = np.where(np.roll(AreaVsMax[:,0],1)!=AreaVsMax[:,0])[0]
BasinData = np.split(AreaVsMax, changes)

# Create DataFrame for analysis with proper basin filtering
df = pd.DataFrame()
CleanBasinData = pd.DataFrame()
valid_sites = []  # Track valid sites

for i, Basin in enumerate(BasinData[1:]):  # Skip first empty basin
    Basin = np.array(Basin)
    Basin = Basin[~np.isnan(Basin).any(axis=1)]
    if len(Basin[:,1]) < 21:  # Keep only the minimum length requirement
        continue
    if i == 0:
        df = pd.DataFrame({str(Basin[0,0]):Basin[:,1]})
        CleanBasinData = pd.DataFrame({str(Basin[0,0]):Basin[:,1]})
        valid_sites.append(str(Basin[0,0]))
    else:
        df1 = pd.DataFrame({str(Basin[0,0]):Basin[:,1]})
        df = pd.concat([df,df1], axis = 1)
        CleanBasinData = pd.concat([CleanBasinData,df1], axis = 1)
        valid_sites.append(str(Basin[0,0]))

# Remove the missing site (1980.65) from valid_sites
valid_sites = [site for site in valid_sites if float(site) != 1980.65]

DrainageAreas = list(df)

###############################################################################
# Section 2: Define Homogeneous Region
###############################################################################

# Calculate L-moments for all sites with sufficient data
rows = []
AllData = np.empty(1)

    #L-moments
for j, k in enumerate(valid_sites): 
    Data = df[k]
    Data = Data.dropna()
    Length = len(Data)
    Datamoms = samlmu(Data,4)
    mom1 = Datamoms[0]
    mom2 = Datamoms[1]
    mom3 = Datamoms[2]
    mom4 = Datamoms[3]
    t = mom2/mom1
    t3 = mom3/mom2
    t4 = mom4/mom2
    
    # Store L-moments and flood values
    DistParams = pelkap(samlmu(Data,4))
    Flood5 = quakap(0.8,DistParams)
    Flood20 = quakap(0.95,DistParams)
    Flood50 = quakap(0.98,DistParams)
    Flood100 = quakap(0.99,DistParams)
    Flood1000 = quakap(0.999,DistParams)
    rows.append([k,Length,mom1,mom2,mom3,mom4,t,t3,t4,Flood5,Flood20,Flood50,Flood100,Flood1000])
    
    DataArray = np.asarray(Data)
    AllData = np.append(AllData,DataArray)

# Create DataFrame of L-moments
LMoments = pd.DataFrame(rows, columns=['Drainage Area','Sample Length','Lmom1','Lmom2','Lmom3','Lmom4',
                                     't','t3','t4','5 Year Flood','20 Year Flood','50 Year Flood',
                                     '100 Year Flood','1000 Year Flood'])

# Calculate mean L-moments for homogeneity assessment
Ubar1 = np.mean(LMoments['t'])
Ubar2 = np.mean(LMoments['t3'])
Ubar3 = np.mean(LMoments['t4'])
UBar = (Ubar1,Ubar2,Ubar3)

# Extract L-moments arrays
lmom1 = np.asarray(LMoments['Lmom1'])
lmom2 = np.asarray(LMoments['Lmom2'])
lmom3 = np.asarray(LMoments['Lmom3'])
lmom4 = np.asarray(LMoments['Lmom4'])
t = np.asarray(LMoments['t'])
t3 = np.asarray(LMoments['t3'])
t4 = np.asarray(LMoments['t4'])
SampleLength = np.asarray(LMoments['Sample Length'])

# Calculate heterogeneity measures
SWorking = []
for l in range(0,18):
    Ul = np.array([t[l],t3[l],t4[l]])
    diff = Ul - UBar
    SWorkingVal = np.outer(diff, diff)  # Create 3x3 matrix for each site
    SWorking.append(SWorkingVal)

S = (1/18)*np.sum(SWorking, axis=0)  # Sum across all sites to get 3x3 covariance matrix

# Calculate discordancy measure
discordant_sites = []
for m in range(0,18):
    Ui = np.array([t[m],t3[m],t4[m]])
    diff = Ui - UBar
    Di = (1/3) * np.dot(np.dot(diff, np.linalg.inv(S)), diff)
    if Di > 3:
        discordant_sites.append(valid_sites[m])
        print(f"Site {valid_sites[m]} is discordant with Di = {Di:.3f}")

# Remove discordant sites from valid_sites
valid_sites = [site for site in valid_sites if site not in discordant_sites]
if discordant_sites:
    print(f"\nRemoved {len(discordant_sites)} discordant site(s) from analysis")

# Filter LMoments DataFrame to match valid sites
LMoments = LMoments[LMoments['Drainage Area'].astype(str).isin(valid_sites)]

# Prepare flood_data dictionary AND finalize valid_sites list based on non-empty data
flood_data = {}
final_valid_sites = [] # Create a new list for sites with actual data
print("\nPopulating flood_data and finalizing valid sites list...")
for site in valid_sites:
    Data = df[site].dropna()
    if Data.empty:
        print(f"Warning: Site {site} has no valid data after dropping NaNs. Excluding from final analysis.")
        continue # Skip this site entirely
    # If data is not empty, add to dict and final list
    flood_data[site] = Data
    final_valid_sites.append(site)

# Overwrite valid_sites with the final list containing only sites with data
valid_sites = final_valid_sites
print(f"Final valid sites count (with data): {len(valid_sites)}.")

# --- Define Return Periods and Probabilities ---
return_periods_prob = {
    '5': 0.8,
    '20': 0.95,
    '50': 0.98,
    '100': 0.99,
    '1000': 0.999
}

# --- Calculate At-Site Quantiles --- 
at_site_quantiles = {}
print("\nCalculating At-Site GEV Quantiles...")
for site in valid_sites:
    site_data = flood_data[site]
    if len(site_data) < 5: # Need a minimum number of points to fit distribution reliably
        print(f" Skipping At-Site fit for site {site} (n={len(site_data)} < 5)")
        continue
    try:
        site_lmoms = samlmu(site_data, 4)
        site_gev_params = pelgev(site_lmoms)
        site_quantiles = {}
        for T, p in return_periods_prob.items(): # Use same probs as regional
            site_quantiles[T] = quagev(p, site_gev_params)
        at_site_quantiles[site] = site_quantiles
    except Exception as e:
        print(f" Error fitting At-Site GEV for site {site}: {e}")
print(f"Calculated At-Site quantiles for {len(at_site_quantiles)} sites.")

# Convert At-Site quantiles dictionary to DataFrame and prepare for comparison
if at_site_quantiles:
    AtSite_df = pd.DataFrame.from_dict(at_site_quantiles, orient='index')
    # Rename columns to simple return periods
    AtSite_df.columns = [str(rp) for rp in return_periods_prob.keys()]

    # Ensure the index is named 'Site ID' (it should be string type from dict keys)
    AtSite_df.index.name = 'Site ID'

    print("\nCreated AtSite_df indexed by Site ID:")
    print(AtSite_df.head())

else:
    print("\nWarning: No At-Site quantiles were calculated. Cannot create AtSite_df.")
    AtSite_df = None

# --- End At-Site Calculation ---

# Calculate heterogeneity measure H1
KapPara = pelkap(samlmu(AllData,4))
local_scales = []
for k in DrainageAreas:
    if k in discordant_sites:  # Skip discordant sites
        continue
    site_data = df[k]
    site_data = site_data.dropna()
    local_scale = np.mean(site_data)
    local_scales.append(local_scale)

LCVBar = np.mean(LMoments['t'])
VWorking = []
for n in range(0,18):
    V1Work = (SampleLength[n])*((t[n]-LCVBar)**2)
    VWorking.append(V1Work)

V1 = sum(VWorking)/sum(LMoments['Sample Length'])

# Monte Carlo simulation for heterogeneity assessment
H1_values = []
for n in range(1,1000):
    simulated_data = []
    for local_scale in local_scales:
        r_site = kappa4.rvs(KapPara[3], KapPara[2], size=500, scale=local_scale)
        simulated_data.extend(r_site)
    
    mu_v = np.std(simulated_data)
    sigma_v = np.mean(simulated_data)
    H1 = ((V1 - mu_v)/sigma_v)
    H1_values.append(H1)

mean_H1 = np.mean(H1_values)

# Assess and report heterogeneity
print("\nHeterogeneity Assessment:")
print(f"Mean H1 value: {mean_H1:.3f}")
print("\nInterpretation:")
if mean_H1 < 1:
    print("Region is acceptably homogeneous (H1 < 1)")
elif mean_H1 < 2:
    print("Region is possibly heterogeneous (1 ≤ H1 < 2)")
else:
    print("Region is definitely heterogeneous (H1 ≥ 2)")

###############################################################################
# Section 3: Fit Regional GEV Distribution using Index-Flood Method (with Box-Cox)
###############################################################################

index_floods = {}
scaled_data_list = []
l_moments_data = {} # Store site-specific L-moments/ratios

print("\nProcessing sites for regional analysis...")
for site in valid_sites:
    Data = df[site].dropna()
    if Data.empty:
        # print(f"Warning: Site {site} has no valid data after dropping NaNs. Skipping.") # Already handled when populating flood_data
        continue

    # Calculate L-moments for the site
    Datamoms = samlmu(Data, 4)
    mom1, mom2, mom3, mom4 = Datamoms[0], Datamoms[1], Datamoms[2], Datamoms[3]

    # Calculate mean as index flood
    index_flood = np.mean(Data)

    # Check if index flood is valid
    if index_flood <= 0:
        print(f"Warning: Site {site} has non-positive index flood (mean = {index_flood:.2f}). Skipping scaling.")
        continue

    index_floods[site] = index_flood
    l_moments_data[site] = {
        'l_moments': {'mom1': mom1, 'mom2': mom2, 'mom3': mom3, 'mom4': mom4},
        'l_ratios': {'t': mom2/mom1 if mom1 else 0, 't3': mom3/mom2 if mom2 else 0, 't4': mom4/mom2 if mom2 else 0}
    }

    # Scale the data by the mean index flood
    scaled_site_data = Data / index_flood
    scaled_data_list.extend(scaled_site_data)

print(f"Collected scaled data from {len(index_floods)} valid sites.")

# Pool scaled data
if not scaled_data_list:
    raise ValueError("No valid scaled data collected. Cannot fit regional distribution.")

pool_scaled_data = np.array(scaled_data_list)

# Apply Box-Cox transformation to pooled scaled data
# Ensure data is positive for Box-Cox
min_pooled = np.min(pool_scaled_data)
if min_pooled <= 0:
    print(f"Warning: Pooled scaled data contains non-positive values (min={min_pooled:.4f}). Adding small shift for Box-Cox.")
    # Add a small shift to make data positive, ensure shift is minimal
    shift = abs(min_pooled) + 1e-9
    pool_scaled_data_shifted = pool_scaled_data + shift
    print(f"Shift added: {shift:.4e}")
else:
    pool_scaled_data_shifted = pool_scaled_data
    shift = 0 # No shift applied

print("\nApplying Box-Cox transformation to pooled scaled data...")
try:
    transformed_pooled_data, pooled_lambda = stats.boxcox(pool_scaled_data_shifted)
    print(f"Box-Cox Lambda (λ) for pooled data: {pooled_lambda:.4f}")
except ValueError as e:
    raise ValueError(f"Box-Cox transformation failed on shifted pooled data: {e}")

# Fit regional distribution to TRANSFORMED pooled scaled data
print(f"\nFitting regional GEV distribution to {len(transformed_pooled_data)} Box-Cox transformed data points...")
regional_lmoms_transformed = samlmu(transformed_pooled_data, 4)
regional_params_transformed = pelgev(regional_lmoms_transformed)  # Use GEV
print(f"Transformed Regional GEV Parameters (xi, alpha, k): {regional_params_transformed}")

# Calculate goodness-of-fit measure (ZDist) in the TRANSFORMED space
print("\nTesting goodness-of-fit of regional GEV distribution (in Box-Cox space...)")

# Calculate regional average L-kurtosis (t4_R) from transformed data
t4_R_transformed = regional_lmoms_transformed[3]/regional_lmoms_transformed[1]  # L4/L2
print(f"Regional average L-kurtosis (t4_R) (Transformed): {t4_R_transformed:.4f}")

# Calculate theoretical L-kurtosis (τ4_DIST) for GEV fitted to transformed data
theoretical_lmoms_transformed = lmrgev(regional_params_transformed, 4)
tau4_DIST_transformed = theoretical_lmoms_transformed[3] # L4/L2
print(f"Theoretical L-kurtosis (τ4_DIST) (Transformed): {tau4_DIST_transformed:.4f}")

# Estimate bias (β4) and standard deviation (σ4) through simulation in TRANSFORMED space
nsim = 1000
t4_sims_transformed = []

print("Running simulations for ZDist bias/std dev in transformed space...")
for i in range(nsim):
    # Generate sample from fitted GEV in TRANSFORMED space
    sim_data_transformed = []
    # We need the sample sizes (n_site) for each site
    for site in index_floods: # Use sites that contributed to scaled data
        n_site = len(flood_data[site]) # Get original sample size
        # Simulate data points from the GEV fitted to the transformed data
        site_sim_transformed = sstats.genextreme.rvs(regional_params_transformed[2], # k
                                         size=n_site,
                                         loc=regional_params_transformed[0],  # xi
                                         scale=regional_params_transformed[1]) # alpha
        sim_data_transformed.extend(site_sim_transformed)

    # Calculate L-kurtosis for simulated TRANSFORMED sample
    if sim_data_transformed:
        sim_lmoms_transformed = samlmu(np.array(sim_data_transformed), 4)
        # Avoid division by zero if L2 is zero or close to it
        if abs(sim_lmoms_transformed[1]) > 1e-9:
             t4_sim = sim_lmoms_transformed[3] / sim_lmoms_transformed[1]
             t4_sims_transformed.append(t4_sim)
        # else: # Optional: handle cases with zero L2 in simulation
        #     print(f"Simulation {i+1}: L2 is near zero, skipping t4 calculation.")
    if (i + 1) % 100 == 0:
        print(f" Simulation {i+1}/{nsim} complete.")

if not t4_sims_transformed:
    raise ValueError("No valid t4 simulations were generated. Check simulation parameters or data.")

t4_sims_transformed = np.array(t4_sims_transformed)
beta4_transformed = np.mean(t4_sims_transformed) - tau4_DIST_transformed  # Bias
sigma4_transformed = np.std(t4_sims_transformed)  # Standard deviation
print(f"Bias (β4) (Transformed): {beta4_transformed:.4f}")
print(f"Standard deviation (σ4) (Transformed): {sigma4_transformed:.4f}")

# Calculate ZDist using transformed values
if sigma4_transformed == 0:
    print("Warning: Standard deviation (σ4) of simulated t4 is zero. Cannot calculate ZDist.")
    Z_DIST_transformed = np.nan
else:
    Z_DIST_transformed = (t4_R_transformed - tau4_DIST_transformed - beta4_transformed) / sigma4_transformed

print(f"\nGoodness-of-fit measure (ZDist) (Transformed Space): {Z_DIST_transformed:.4f}")
print("Interpretation:")
if abs(Z_DIST_transformed) <= 1.64:
    print("The GEV distribution appears suitable in the Box-Cox transformed space (ZDist| ≤ 1.64)")
else:
    print("The GEV distribution may not be suitable in the Box-Cox transformed space (|ZDist| > 1.64)")

# Calculate regional growth factors in TRANSFORMED space
regional_growth_factors_transformed = {}
for T, p in return_periods_prob.items(): # Use the globally defined dict
    regional_growth_factors_transformed[T] = quagev(p, regional_params_transformed)
print(f"\nRegional Growth Factors (Transformed Space): {regional_growth_factors_transformed}")

# Calculate flood quantiles for each gauged site using the regional method
flood_values = {} # This will hold the final UNTRANSFORMED QT values for regression
print("\nCalculating final flood quantiles using inverse Box-Cox...")
for site in index_floods: # Iterate through sites that had valid index flood
    index_flood = index_floods[site]
    site_l_data = l_moments_data[site]
    site_flood_quantiles = {}

    for T, transformed_gf in regional_growth_factors_transformed.items():
        # Inverse transform the growth factor using the pooled lambda
        # inv_boxcox(transformed_value, lambda) - shift
        untransformed_gf = inv_boxcox(transformed_gf, pooled_lambda) - shift

        # Multiply by the site's index flood (mean)
        site_flood_quantiles[T] = index_flood * untransformed_gf

    flood_values[site] = {
        '5': site_flood_quantiles['5'],
        '20': site_flood_quantiles['20'],
        '50': site_flood_quantiles['50'],
        '100': site_flood_quantiles['100'],
        '1000': site_flood_quantiles['1000'],
        # Include original L-moments/ratios for potential use/plotting
        'l_moments': site_l_data['l_moments'],
        'l_ratios': site_l_data['l_ratios']
    }

# Rebuild LMoments DataFrame with regionally derived (and untransformed) quantiles for regression input
rows_regional = []
for site in valid_sites:
    if site in flood_values: # Check if site was processed and has values
        site_vals = flood_values[site]
        site_l_moms = site_vals['l_moments'] # These are original L-moments
        site_l_rats = site_vals['l_ratios'] # These are original L-ratios
        original_length = len(df[site].dropna())

        rows_regional.append([
            site, original_length,
            site_l_moms['mom1'], site_l_moms['mom2'], site_l_moms['mom3'], site_l_moms['mom4'],
            site_l_rats['t'], site_l_rats['t3'], site_l_rats['t4'],
            site_vals['5'], site_vals['20'], site_vals['50'], site_vals['100'], site_vals['1000']
        ])

# Filter out rows with NaN flood values which can occur if inverse transform fails
LMoments_regional = pd.DataFrame(rows_regional, columns=['Drainage Area','Sample Length','Lmom1','Lmom2','Lmom3','Lmom4',
                                             't','t3','t4','5 Year Flood','20 Year Flood','50 Year Flood',
                                             '100 Year Flood','1000 Year Flood'])
LMoments_regional.dropna(subset=['5 Year Flood', '20 Year Flood', '50 Year Flood', '100 Year Flood', '1000 Year Flood'], inplace=True)

# Ensure Drainage Area is string for consistent filtering later
LMoments_regional['Drainage Area'] = LMoments_regional['Drainage Area'].astype(str)

print("Rebuilt LMoments DataFrame with final regionally derived flood quantiles.")

###############################################################################
# Section 4: Form Regression Model for Ungauged Sites
###############################################################################

# Reload CompiledStats
CompiledStats = pd.read_excel('GaugingLocations.xlsx', index_col=None, header=0)

if LMoments_regional.empty: raise ValueError("LMoments_regional is empty.")
final_regression_sites = LMoments_regional['Drainage Area'].tolist()

# Filter CompiledStats based on sites in LMoments_regional BEFORE dropping columns
Xvar = CompiledStats[CompiledStats['Drainage Area (WRA) (km^2)'].astype(str).isin(final_regression_sites)].copy()

# Align LMoments_regional to match the order of Xvar (based on Drainage Area)
Xvar_index_col = 'Drainage Area (WRA) (km^2)'
Xvar = Xvar.set_index(Xvar[Xvar_index_col].astype(str))
LMoments_aligned = LMoments_regional.set_index('Drainage Area').loc[Xvar.index] # Align using index

# --- Calculate DAMax BEFORE dropping the column ---
DA_col_name = 'Drainage Area (WRA) (km^2)'
if DA_col_name in Xvar.columns:
    DA = Xvar[DA_col_name]
    DAMax = DA.max() + 1e-9 # Add epsilon for safety
    print(f"Calculated DAMax: {DAMax:.4f}")
else:
    raise ValueError(f"Column '{DA_col_name}' not found in Xvar for calculating DAMax.")
# --- End DAMax Calculation ---

# Prepare dependent variables from the ALIGNED DataFrame
Yvar5 = LMoments_aligned['5 Year Flood']
Yvar20 = LMoments_aligned['20 Year Flood']
Yvar50 = LMoments_aligned['50 Year Flood']
Yvar100 = LMoments_aligned['100 Year Flood']
Yvar1000 = LMoments_aligned['1000 Year Flood']

# Reset Xvar index before dropping columns
Xvar.reset_index(drop=True, inplace=True)

if Xvar.empty or len(Xvar) != len(Yvar5): raise ValueError(f"Mismatch Xvar ({len(Xvar)})/Yvar ({len(Yvar5)}) after alignment")

# Modify drop_cols to KEEP Drainage Area
drop_cols = ['Channel Width (m)', 'Gauge Label', 'River Name', 'Location Name', 'River #', 'Location #',
             'Notes', 'Drainage Area (Polygon) (m^2)', 'Drainage Area (Polygon) (km^2)',
             'Area Error (Percent)', 'Dist North (km)', 't', 't3', 't4', '1 Year Flood',
             '5 Year Flood', '20 Year Flood', '50 Year Flood', '100 Year Flood', '1000 Year Flood']
             # Removed DA_col_name from here
Xvar.drop(columns=[col for col in drop_cols if col in Xvar.columns], inplace=True)
print(f"Xvar columns after dropping: {Xvar.columns.tolist()}") # Debug print

# --- Normalization Calculation and Application ---
print("\nNormalizing independent variables (Xvar)...")
# Ensure columns exist (now includes DA)
required_cols = ['E - TWD97', 'N - TWD97', 'Drainage Area (WRA) (km^2)', 'Mean HS Angle (Deg)',
                 'Max HS Angle (Deg)', 'Std Dev HS Angle (Deg)', 'Mean Annual Precip']
if not all(col in Xvar.columns for col in required_cols):
    raise ValueError(f"Missing required columns in Xvar for normalization. Found: {Xvar.columns.tolist()}")

E_m = Xvar['E - TWD97']
N_m = Xvar['N - TWD97']
DA = Xvar['Drainage Area (WRA) (km^2)'] # Get DA column again for normalization
MeanHS = Xvar['Mean HS Angle (Deg)']
MaxHS = Xvar['Max HS Angle (Deg)']
STDHS = Xvar['Std Dev HS Angle (Deg)']
MAP = Xvar['Mean Annual Precip']

# Calculate normalization factors (Max values)
epsilon = 1e-9
E_mMax = E_m.max() + epsilon
N_mMax = N_m.max() + epsilon
# DAMax was calculated before
MeanHSMax = MeanHS.max() + epsilon
MaxHSMax = MaxHS.max() + epsilon
STDHSMax = STDHS.max() + epsilon
MAPMax = MAP.max() + epsilon

# Store factors in a dictionary
norm_factors = {
    'E_mMax': E_mMax, 'N_mMax': N_mMax, 'DAMax': DAMax,
    'MeanHSMax': MeanHSMax, 'MaxHSMax': MaxHSMax, 'STDHSMax': STDHSMax, 'MAPMax': MAPMax
}

# Apply normalization to the Xvar DataFrame (including DA)
Xvar['E - TWD97'] = E_m / norm_factors['E_mMax']
Xvar['N - TWD97'] = N_m / norm_factors['N_mMax']
Xvar['Drainage Area (WRA) (km^2)'] = DA / norm_factors['DAMax'] # Normalize DA
Xvar['Mean HS Angle (Deg)'] = MeanHS / norm_factors['MeanHSMax']
Xvar['Max HS Angle (Deg)'] = MaxHS / norm_factors['MaxHSMax']
Xvar['Std Dev HS Angle (Deg)'] = STDHS / norm_factors['STDHSMax']
Xvar['Mean Annual Precip'] = MAP / norm_factors['MAPMax']
print("Normalization applied.")
# --- End Normalization ---

Xvar_array = Xvar.values # Convert FINAL NORMALIZED Xvar (now 7 features) to numpy array

# --- Add back Regression Functions and Training ---
# Define regression functions
def hypothesis(theta, Xvar_a): # Use different var name to avoid scope issues
    """Linear regression model"""
    # Add bias term (column of ones) directly here
    X_with_bias = np.column_stack([np.ones(len(Xvar_a)), Xvar_a])
    return np.dot(X_with_bias, theta)

def computeCost(Xvar_a, y, theta):
    """Compute cost (using hypothesis function)"""
    y1 = hypothesis(theta, Xvar_a)
    m = len(Xvar_a)
    if m == 0: return 0
    # Using Mean Squared Error cost J = (1/(2*m)) * sum((y1-y)^2)
    # Or Root Mean Squared Error: sqrt(sum((y1-y)^2)/m)
    # Let's stick to a common cost function for gradient descent, like MSE variant
    return np.sum((y1-y)**2)/(2*m)

def gradientDescent(Xvar_a, y, theta_init, alpha, max_iterations):
    """Gradient descent to find optimal theta"""
    theta = theta_init.copy() # Work on a copy
    m = len(Xvar_a)
    if m == 0: return [], 0, theta # Handle empty input
    J_history = []
    X_with_bias = np.column_stack([np.ones(m), Xvar_a])
    y = np.array(y) # Ensure y is a numpy array

    for iteration in range(max_iterations):
        y_pred = np.dot(X_with_bias, theta)
        error = y_pred - y
        gradient = (1/m) * np.dot(X_with_bias.T, error)
        theta = theta - alpha * gradient

        cost = computeCost(Xvar_a, y, theta)
        J_history.append(cost)

        # Convergence check (optional but recommended)
        if iteration > 0 and abs(J_history[-1] - J_history[-2]) < 1e-7:
            print(f" Gradient descent converged at iteration {iteration+1}")
            break

    final_cost = J_history[-1] if J_history else 0
    return J_history, final_cost, theta

# Optimize regression parameters for each return period
theta_initial = np.zeros(Xvar_array.shape[1] + 1)  # Initialize theta (+1 for bias term)
print("\nOptimizing theta values for each return period using Gradient Descent...")

# 5-year flood
J5, j5, Theta5 = gradientDescent(Xvar_array, Yvar5, theta_initial, 0.01, 20000)
print(f"5-Year Theta optimized. Final cost: {j5:.4f}")
y_hat5 = hypothesis(Theta5, Xvar_array) # Calculate predictions

# 20-year flood
J20, j20, Theta20 = gradientDescent(Xvar_array, Yvar20, theta_initial, 0.01, 20000)
print(f"20-Year Theta optimized. Final cost: {j20:.4f}")
y_hat20 = hypothesis(Theta20, Xvar_array) # Calculate predictions

# 50-year flood
J50, j50, Theta50 = gradientDescent(Xvar_array, Yvar50, theta_initial, 0.01, 20000)
print(f"50-Year Theta optimized. Final cost: {j50:.4f}")
y_hat50 = hypothesis(Theta50, Xvar_array) # Calculate predictions

# 100-year flood
J100, j100, Theta100 = gradientDescent(Xvar_array, Yvar100, theta_initial, 0.01, 20000)
print(f"100-Year Theta optimized. Final cost: {j100:.4f}")
y_hat100 = hypothesis(Theta100, Xvar_array) # Calculate predictions

# 1000-year flood
J1000, j1000, Theta1000 = gradientDescent(Xvar_array, Yvar1000, theta_initial, 0.01, 20000)
print(f"1000-Year Theta optimized. Final cost: {j1000:.4f}")
y_hat1000 = hypothesis(Theta1000, Xvar_array) # Calculate predictions

# --- Store y_hat predictions in a dictionary ---
y_hat_predictions = {
    '5': y_hat5,
    '20': y_hat20,
    '50': y_hat50,
    '100': y_hat100,
    '1000': y_hat1000
}

# --- End Regression Training ---

###############################################################################
# Section 5: Predict Flood Values for Ungauged Sites
###############################################################################


def predict_floods(site_X_normalized, theta5, theta20, theta50, theta100, theta1000):
    """Predicts flood values for given normalized site data and trained thetas."""
    # Add bias term (1) to the site data
    X_with_bias = np.insert(site_X_normalized, 0, 1)
    return np.array([
        np.dot(X_with_bias, theta5),
        np.dot(X_with_bias, theta20),
        np.dot(X_with_bias, theta50),
        np.dot(X_with_bias, theta100),
        np.dot(X_with_bias, theta1000)
    ])

# Prepare NORMALIZED data for ungauged sites using the STORED factors
Chin_X_norm = np.array([
    241959.07/norm_factors['E_mMax'],    # Easting
    2492450.44/norm_factors['N_mMax'],   # Northing
    135.42/norm_factors['DAMax'],       # Drainage Area
    29.51/norm_factors['MeanHSMax'],   # Mean HS
    62.21/norm_factors['MaxHSMax'],    # Max HS
    10.90/norm_factors['STDHSMax'],    # Std Dev HS
    3592/norm_factors['MAPMax']        # MAP
])
Sand_X_norm = np.array([
    212975.30/norm_factors['E_mMax'],
    2512663.00/norm_factors['N_mMax'],
    408.51/norm_factors['DAMax'],       # Drainage Area
    29.88/norm_factors['MeanHSMax'],
    62.22/norm_factors['MaxHSMax'],
    10.78/norm_factors['STDHSMax'],
    4016/norm_factors['MAPMax']
])
Dahou_X_norm = np.array([
    216150.313/norm_factors['E_mMax'],
    2495692.729/norm_factors['N_mMax'],
    55.93/norm_factors['DAMax'],        # Drainage Area
    30.47/norm_factors['MeanHSMax'],
    75.05/norm_factors['MaxHSMax'],
    11.035/norm_factors['STDHSMax'],
    4215/norm_factors['MAPMax']
])
Liji_X_norm = np.array([
    255740.876/norm_factors['E_mMax'],
    2517702/norm_factors['N_mMax'],
    148.57/norm_factors['DAMax'],       # Drainage Area
    29.65/norm_factors['MeanHSMax'],
    75.10/norm_factors['MaxHSMax'],
    10.44/norm_factors['STDHSMax'],
    2687/norm_factors['MAPMax']
])

# Calculate predictions for ungauged sites
Chin_Floods = predict_floods(Chin_X_norm, Theta5, Theta20, Theta50, Theta100, Theta1000)
Sand_Floods = predict_floods(Sand_X_norm, Theta5, Theta20, Theta50, Theta100, Theta1000)
Dahou_Floods = predict_floods(Dahou_X_norm, Theta5, Theta20, Theta50, Theta100, Theta1000)
Liji_Floods = predict_floods(Liji_X_norm, Theta5, Theta20, Theta50, Theta100, Theta1000)

# Print results
print("\nPredicted Flood Values for Ungauged Sites (m³/s):")
print("Return Period (years)     5      20     50     100    1000")
print(f"Chin Creek:         {Chin_Floods[0]:>6.0f} {Chin_Floods[1]:>6.0f} {Chin_Floods[2]:>6.0f} {Chin_Floods[3]:>6.0f} {Chin_Floods[4]:>6.0f}")
print(f"Sand Creek:         {Sand_Floods[0]:>6.0f} {Sand_Floods[1]:>6.0f} {Sand_Floods[2]:>6.0f} {Sand_Floods[3]:>6.0f} {Sand_Floods[4]:>6.0f}")
print(f"Dahou Creek:        {Dahou_Floods[0]:>6.0f} {Dahou_Floods[1]:>6.0f} {Dahou_Floods[2]:>6.0f} {Dahou_Floods[3]:>6.0f} {Dahou_Floods[4]:>6.0f}")
print(f"Liji Creek:         {Liji_Floods[0]:>6.0f} {Liji_Floods[1]:>6.0f} {Liji_Floods[2]:>6.0f} {Liji_Floods[3]:>6.0f} {Liji_Floods[4]:>6.0f}")


###############################################################################
# Section 6: Visualization
###############################################################################

def calculate_r2(y_true, y_pred):
    """Calculates R-squared value."""
    # Handle potential NaNs or Infs resulting from calculations/joins
    valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]
    if len(y_true_clean) < 2: # Cannot compute R2 with fewer than 2 points
        return np.nan
    ss_res = np.sum((y_true_clean - y_pred_clean)**2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean))**2)
    if ss_tot == 0: # Avoid division by zero if all y_true values are the same
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

def plot_distribution_fit_boxcox(valid_sites, flood_data, index_floods, l_moments_data,
                                 pooled_lambda, shift, transformed_pooled_data,
                                 regional_params_transformed, regional_growth_factors_transformed):
    """Plot Box-Cox transformed scaled data against the GEV fit in transformed space."""
    print("\nGenerating Transformed GEV Fit Plots (Per Site)...")

    # Use the final list of sites that have flood_data
    sites_to_plot = list(flood_data.keys())
    n_sites = len(sites_to_plot)
    if n_sites == 0:
        print("No valid sites with data found for plotting GEV fits.")
        return

    n_cols = 4
    n_rows = math.ceil(n_sites / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), squeeze=False)
    title_str = f'Regional GEV Fit in Box-Cox Transformed Space (λ={pooled_lambda:.2f})'
    fig.suptitle(title_str, fontsize=16)
    axs = axs.ravel()

    # Theoretical curve points
    theo_probs = np.linspace(0.01, 0.99, 100)
    theo_return_periods = 1 / (1 - theo_probs)
    theo_quantiles_transformed = [quagev(p, regional_params_transformed) for p in theo_probs]

    # Design points (transformed)
    analysis_T_transformed = list(regional_growth_factors_transformed.keys())
    analysis_GF_transformed_vals = list(regional_growth_factors_transformed.values())
    analysis_T_numeric = [int(T) for T in analysis_T_transformed]

    plot_count = 0
    for site in sites_to_plot:
        ax = axs[plot_count]
        if site not in flood_data or site not in index_floods:
            ax.set_title(f"Site {site} - Missing Data")
            plot_count += 1
            continue

        # Get original data and index flood
        Data = flood_data[site]
        index_flood = index_floods[site]

        # Scale original data
        scaled_data = Data / index_flood

        # Transform scaled data using the POOLED lambda and shift
        try:
            scaled_data_shifted = scaled_data + shift
            if np.any(scaled_data_shifted <= 0):
                 print(f" Site {site}: Scaled+shifted data still non-positive. Skipping BoxCox.")
                 ax.set_title(f"Site {site}\nNon-positive after shift")
                 plot_count += 1
                 continue
            transformed_scaled_data = stats.boxcox(scaled_data_shifted, lmbda=pooled_lambda)
        except ValueError as e:
            print(f" Site {site}: Error applying BoxCox with pooled lambda: {e}")
            ax.set_title(f"Site {site}\nBoxCox Error")
            plot_count += 1
            continue

        # Calculate empirical plotting positions for TRANSFORMED data
        sorted_transformed_data = np.sort(transformed_scaled_data)
        n = len(sorted_transformed_data)
        emp_probs = np.arange(1, n + 1) / (n + 1)
        emp_return_periods = 1 / (1 - emp_probs)

        # Plot TRANSFORMED empirical data
        ax.semilogx(emp_return_periods, sorted_transformed_data, 'bo',
                    markersize=4, label='Observed (Scaled & Transformed)')
        # Plot theoretical regional GEV curve (transformed space)
        ax.semilogx(theo_return_periods, theo_quantiles_transformed, 'r-',
                    label='Regional GEV (Transformed)')
        # Plot design points (transformed space)
        ax.semilogx(analysis_T_numeric, analysis_GF_transformed_vals, 'k*',
                    markersize=10, label='Design Growth Factors (Transformed)')

        # Add title with original L-moment ratios for reference
        if site in l_moments_data:
            l_ratios = l_moments_data[site]['l_ratios']
            site_title = f'Site {site}'
            site_title += f'\nOriginal L-ratios: t={l_ratios.get("t", 0):.3f}, t3={l_ratios.get("t3", 0):.3f}, t4={l_ratios.get("t4", 0):.3f}'
            ax.set_title(site_title, fontsize=10)
        else:
            ax.set_title(f'Site {site}', fontsize=10)

        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('Transformed Growth Factor')
        ax.grid(True)
        ax.legend(fontsize='small')
        plot_count += 1

    # Hide any unused subplots
    for j in range(plot_count, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('regional_gev_fit_transformed_sites.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- Function to Plot Regression Performance ---
def plot_regression_performance(Yvars, Y_hats, return_periods):
    """Generates scatter plots comparing regional estimates vs regression predictions."""
    print("\nGenerating Regression Performance Plots...")
    num_plots = len(return_periods)
    if num_plots == 0:
        return

    n_cols = 3
    n_rows = math.ceil(num_plots / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axs = axs.ravel()
    fig.suptitle('Regression Performance: Regional Estimate vs. Regression Prediction (Gauged Sites)', fontsize=14)

    plot_idx = 0
    for T, y_true, y_pred in zip(return_periods, Yvars, Y_hats):
        if y_true is None or y_pred is None:
            print(f"Skipping {T}-year plot due to missing data.")
            continue

        ax = axs[plot_idx]
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='Gauged Sites')

        # Add 1:1 line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

        # Calculate R-squared
        r2 = calculate_r2(y_true, y_pred)

        ax.set_title(f'{T}-Year Flood (R² = {r2:.3f})')
        ax.set_xlabel('Regional Estimate (Index Flood * GF)')
        ax.set_ylabel('Regression Prediction')
        ax.grid(True)
        ax.legend()
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('regression_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
# --- END Regression Plot Function ---

# --- ADD Function to Plot At-Site vs Regression Estimates ---
def plot_atsite_vs_regression(AtSite_df, Regression_Predictions_df, return_periods):
    """Generates scatter plots comparing At-Site estimates vs RFA Regression predictions."""
    print("\nGenerating At-Site vs. Regression Prediction Plots...")

    # Find common sites (assuming indices are correctly set before function call)
    common_sites = AtSite_df.index.intersection(Regression_Predictions_df.index)
    if len(common_sites) == 0:
        print("Warning: No common sites found between At-Site and Regression predictions. Skipping plots.")
        return

    AtSite_common = AtSite_df.loc[common_sites]
    Regression_common = Regression_Predictions_df.loc[common_sites]

    num_plots = len(return_periods)
    n_cols = 3
    n_rows = math.ceil(num_plots / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axs = axs.ravel()
    fig.suptitle('Comparison: At-Site GEV Estimate vs. RFA Regression Prediction', fontsize=14)

    plot_idx = 0
    for T in return_periods:
        y_at_site = AtSite_common[T] # At-site estimate for return period T
        y_regression = Regression_common[f'{T}_hat'] # Regression prediction for T (needs matching column name)

        # Drop NaNs that might exist in either column for this specific T
        valid_comparison = pd.DataFrame({'AtSite': y_at_site, 'Regression': y_regression}).dropna()
        if valid_comparison.empty:
            print(f" Skipping {T}-year plot due to missing data after alignment.")
            continue

        y_at_site_plot = valid_comparison['AtSite']
        y_regression_plot = valid_comparison['Regression']

        ax = axs[plot_idx]
        ax.scatter(y_at_site_plot, y_regression_plot, alpha=0.7, edgecolors='k', label='Gauged Sites')

        # Add 1:1 line
        min_val = min(min(y_at_site_plot), min(y_regression_plot))
        max_val = max(max(y_at_site_plot), max(y_regression_plot))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

        # Calculate R-squared
        r2 = calculate_r2(y_at_site_plot, y_regression_plot)

        ax.set_title(f'{T}-Year Flood (R² = {r2:.3f})')
        ax.set_xlabel('At-Site GEV Estimate')
        ax.set_ylabel('RFA Regression Prediction') # Updated Label
        ax.grid(True)
        ax.legend()
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('atsite_vs_regression_comparison.png', dpi=300, bbox_inches='tight') # Updated filename
    plt.show()
# --- END At-Site vs Regression Plot Function ---

# --- ADD Function to Plot Pooled GEV Fit ---
def plot_pooled_gev_fit_boxcox(transformed_pooled_data, pooled_lambda,
                               regional_params_transformed,
                               regional_growth_factors_transformed,
                               return_periods_prob):
    """Plots the regional GEV fit against the pooled, scaled, Box-Cox transformed data."""
    print("\nGenerating Pooled GEV Fit Plot (Transformed Space)...")

    if transformed_pooled_data is None or len(transformed_pooled_data) == 0:
        print(" No pooled transformed data available to plot.")
        return

    # Calculate empirical plotting positions for the POOLED data
    sorted_transformed_data = np.sort(transformed_pooled_data)
    n = len(sorted_transformed_data)
    emp_probs = np.arange(1, n + 1) / (n + 1)
    emp_return_periods = 1 / (1 - emp_probs)

    # Theoretical curve points (same as in the per-site plot)
    theo_probs = np.linspace(0.01, 0.99, 200) # More points for smoother curve
    theo_return_periods = 1 / (1 - theo_probs)
    theo_quantiles_transformed = [quagev(p, regional_params_transformed) for p in theo_probs]

    # Design points (transformed)
    analysis_T_transformed = list(regional_growth_factors_transformed.keys())
    analysis_GF_transformed_vals = list(regional_growth_factors_transformed.values())
    analysis_T_numeric = [int(T) for T in analysis_T_transformed]

    # Plotting
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    # Plot empirical POOLED data
    ax.semilogx(emp_return_periods, sorted_transformed_data, 'bo',
                markersize=3, alpha=0.6, label=f'Pooled Data Points (n={n})')
    # Plot theoretical regional GEV curve (transformed space)
    ax.semilogx(theo_return_periods, theo_quantiles_transformed, 'r-', linewidth=2,
                label='Regional GEV Fit (Transformed)')
    # Plot design points (transformed space)
    ax.semilogx(analysis_T_numeric, analysis_GF_transformed_vals, 'k*',
                markersize=10, label='Design Growth Factors (Transformed)')

    title_str = f'Regional GEV Distribution Fit to Pooled Data in Box-Cox Transformed Space (λ={pooled_lambda:.3f})'
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('Transformed Growth Factor')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize='medium')

    plt.tight_layout()
    plt.savefig('regional_gev_fit_pooled_transformed.png', dpi=300, bbox_inches='tight')
    plt.savefig('regional_gev_fit_pooled_transformed_SVG.svg', bbox_inches='tight')
    print(" Saved pooled GEV fit plot to regional_gev_fit_pooled_transformed.png and regional_gev_fit_pooled_transformed.svg")
    plt.show()
# --- END Pooled GEV Plot Function ---

###############################################################################
# Section 7: Leave-One-Out Cross-Validation (LOOCV)
###############################################################################

# --- Prepare data required for LOOCV (using original UNNORMALIZED X and Regional Y) ---
print("\n--- Preparing data for LOOCV --- ")
try:
    # 1. Prepare Yvars dictionary from the aligned regional estimates
    if 'LMoments_aligned' in locals() and 'return_periods_prob' in locals():
        Yvars = {}
        for T_str, _ in return_periods_prob.items():
            col_name = f'{T_str} Year Flood'
            if col_name in LMoments_aligned.columns:
                Yvars[T_str] = LMoments_aligned[col_name]
            else:
                raise ValueError(f"Column '{col_name}' not found in LMoments_aligned for Yvars.")
        print(f"  Created Yvars dictionary for LOOCV with keys: {list(Yvars.keys())}")
    else:
        raise NameError("LMoments_aligned or return_periods_prob not defined before LOOCV prep.")

    # 2. Prepare aligned, UNNORMALIZED X predictor variables
    if 'CompiledStats' in locals() and 'LMoments_aligned' in locals():
        required_cols = ['E - TWD97', 'N - TWD97', 'Drainage Area (WRA) (km^2)', 'Mean HS Angle (Deg)',
                         'Max HS Angle (Deg)', 'Std Dev HS Angle (Deg)', 'Mean Annual Precip']
        da_col_name = 'Drainage Area (WRA) (km^2)'

        # Filter original CompiledStats by the sites present in the final aligned L-Moments
        sites_for_loocv = LMoments_aligned.index.astype(str).tolist()
        Xvar_unnormalized_all_cols = CompiledStats[CompiledStats[da_col_name].astype(str).isin(sites_for_loocv)].copy()

        # Select only the required predictor columns
        missing_cols = [col for col in required_cols if col not in Xvar_unnormalized_all_cols.columns]
        if missing_cols:
            raise ValueError(f"Missing required predictor columns in CompiledStats: {missing_cols}")
        Xvar_unnormalized_final_temp = Xvar_unnormalized_all_cols[required_cols]

        # Set index to DA (as string) for alignment
        Xvar_unnormalized_final_temp[da_col_name] = Xvar_unnormalized_final_temp[da_col_name].astype(str)
        Xvar_unnormalized_final_temp = Xvar_unnormalized_final_temp.set_index(da_col_name)

        # Reindex to match the exact order of LMoments_aligned/Yvars
        Xvar_unnormalized_final = Xvar_unnormalized_final_temp.loc[LMoments_aligned.index]
        print(f"  Created aligned, unnormalized Xvar_unnormalized_final for LOOCV, shape: {Xvar_unnormalized_final.shape}")

    else:
        raise NameError("CompiledStats or LMoments_aligned not defined before LOOCV prep.")

    # 3. Define site IDs and Yvars structure for the function
    site_ids_for_loocv = Xvar_unnormalized_final.index.tolist() # Use index from the prepared Xvar
    Yvars_for_loocv = {rp: Yvars[rp].values for rp in return_periods_prob.keys()} # Convert Series to numpy arrays

except Exception as e:
    print(f"Error preparing data for LOOCV: {e}")
    # Set variables to None or empty to prevent NameError in the next block
    Xvar_unnormalized_final = None
    Yvars_for_loocv = None
    site_ids_for_loocv = None
# --- End LOOCV Data Prep ---

def predict_one(site_X_normalized, theta):
    """Predicts flood value for a single site given normalized data and theta."""
    site_X_normalized = np.array(site_X_normalized).flatten()
    X_with_bias = np.insert(site_X_normalized, 0, 1)
    if X_with_bias.shape[0] != theta.shape[0]:
        raise ValueError(f"Shape mismatch in predict_one: X {X_with_bias.shape}, theta {theta.shape}")
    return np.dot(X_with_bias, theta)

def plot_loocv_results(atsite_df, predicted_df, return_periods):
    """Generates scatter plots comparing At-Site GEV estimates vs LOOCV predictions."""
    print("\nGenerating LOOCV vs At-Site Performance Plots...") # Updated print
    if atsite_df.empty or predicted_df.empty: return

    # DataFrames should already be aligned by perform_loocv

    num_plots = len(return_periods)
    n_cols = 3; n_rows = math.ceil(num_plots / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axs = axs.ravel()
    fig.suptitle('LOOCV Performance: At-Site GEV Estimate vs. LOOCV Regression Prediction', fontsize=14) # Updated title
    plot_idx = 0
    for T in return_periods:
        # Column names in atsite_df are '5', '20', etc.
        # Column names in predicted_df are '5_cv_pred', '20_cv_pred', etc.
        y_true_at_site = atsite_df[T]
        y_pred_loocv = predicted_df[f'{T}_cv_pred']

        valid_comp = pd.DataFrame({'AtSite': y_true_at_site, 'LOOCV_Pred': y_pred_loocv}).dropna()
        if valid_comp.empty: continue

        y_true_plot = valid_comp['AtSite']; y_pred_plot = valid_comp['LOOCV_Pred']

        ax = axs[plot_idx]
        ax.scatter(y_true_plot, y_pred_plot, alpha=0.7, edgecolors='k', label='Gauged Sites (LOOCV)')
        min_val = min(min(y_true_plot), min(y_pred_plot)); max_val = max(max(y_true_plot), max(y_pred_plot))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
        r2 = calculate_r2(y_true_plot, y_pred_plot)
        rmse = np.sqrt(np.mean((y_true_plot - y_pred_plot)**2))
        ax.set_title(f'{T}-Year Flood (R²={r2:.3f}, RMSE={rmse:.1f})')
        ax.set_xlabel('At-Site GEV Estimate') # Updated X label
        ax.set_ylabel('LOOCV Regression Prediction')
        ax.grid(True); ax.legend(); plot_idx += 1
    for i in range(plot_idx, len(axs)): fig.delaxes(axs[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('loocv_vs_atsite_performance.png', dpi=300, bbox_inches='tight') # Updated filename
    plt.show()

# Modify perform_loocv to accept AtSite_df and perform the correct comparison
def perform_loocv(Xvar_full, Yvars_dict, site_ids, AtSite_df_full):
    """Performs LOOCV and compares predictions against At-Site estimates."""
    print("\nStarting Leave-One-Out Cross-Validation (vs At-Site Estimates)...") # Updated print
    n_sites = len(site_ids); loocv_predictions = {site_id: {} for site_id in site_ids}
    return_periods = list(Yvars_dict.keys())
    if n_sites == 0: return None

    # --- LOOCV Loop (Training/Prediction part unchanged) ---
    for i, held_out_site_id in enumerate(site_ids):
        print(f" LOOCV Fold {i+1}/{n_sites} (Holding out: {held_out_site_id})")
        train_indices = [idx for idx, site in enumerate(site_ids) if site != held_out_site_id]
        test_index = i
        X_train = Xvar_full.iloc[train_indices]; X_test_single = Xvar_full.iloc[test_index]
        Y_train = {T: Yvars_dict[T][train_indices] for T in return_periods} # Train using Regional Estimates

        # Normalize WITHIN the fold (based on training data)
        train_norm_factors = {}; epsilon = 1e-9
        for col in X_train.columns: train_norm_factors[f'{col}_Max'] = X_train[col].max() + epsilon
        X_train_norm = X_train.copy()
        for col in X_train_norm.columns: X_train_norm[col] = X_train[col] / train_norm_factors[f'{col}_Max']
        X_test_norm = X_test_single.copy()
        for col in X_test_norm.index: X_test_norm[col] = X_test_single[col] / train_norm_factors[f'{col}_Max']

        X_train_array = X_train_norm.values; X_test_array = X_test_norm.values
        initial_theta = np.zeros(X_train_array.shape[1] + 1)

        for T in return_periods:
            _, _, Theta_T_cv = gradientDescent(X_train_array, Y_train[T], initial_theta, 0.01, 20000)
            y_pred_single = predict_one(X_test_array, Theta_T_cv)
            loocv_predictions[held_out_site_id][T] = y_pred_single
    # --- End LOOCV Loop ---

    # --- Process LOOCV Results vs At-Site Estimates ---
    Pred_df = pd.DataFrame.from_dict(loocv_predictions, orient='index')
    Pred_df.columns = [f'{T}_cv_pred' for T in return_periods]
    Pred_df.index.name = 'Site ID'

    # Align At-Site and Predicted DataFrames based on their indices (Site ID)
    # AtSite_df_full should already have 'Site ID' as index name (ensured before calling)
    common_idx = AtSite_df_full.index.intersection(Pred_df.index)
    if common_idx.empty:
        print(" Warning: No common Site IDs found between AtSite_df_full and Pred_df inside perform_loocv. Check input data alignment.")
        return None, None, None # Return None if no common data

    AtSite_aligned = AtSite_df_full.loc[common_idx]
    Pred_aligned = Pred_df.loc[common_idx]

    print("\nLOOCV vs At-Site Performance Metrics:") # Updated print
    metrics = {}
    for T in return_periods:
        # Use At-Site values as the 'true' value for comparison
        y_true = AtSite_aligned[T]
        y_pred = Pred_aligned[f'{T}_cv_pred']
        r2 = calculate_r2(y_true, y_pred);
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        metrics[T] = {'R2': r2, 'RMSE': rmse}
        print(f"  {T}-Year: R² = {r2:.4f}, RMSE = {rmse:.2f}")

    # Plot results comparing At-Site vs LOOCV Prediction
    plot_loocv_results(AtSite_aligned, Pred_aligned, return_periods)

    return metrics, AtSite_aligned, Pred_aligned
# --- END LOOCV Section ---


print("\nRunning Final Analysis Steps...")

# 1. Plot Transformed GEV Fit
try:
    plot_distribution_fit_boxcox(valid_sites, flood_data, index_floods, l_moments_data,
                                 pooled_lambda, shift, transformed_pooled_data,
                                 regional_params_transformed, regional_growth_factors_transformed)
except NameError as e:
    print(f"Could not generate GEV fit plot. Missing variable: {e}")
except Exception as e:
    print(f"Error generating GEV fit plot: {e}")

# 2. Plot Pooled GEV Fit
print("\n--- Generating Pooled GEV Fit Plot ---")
try:
    # Check necessary variables exist
    if ('transformed_pooled_data' in locals() and
        'pooled_lambda' in locals() and
        'regional_params_transformed' in locals() and
        'regional_growth_factors_transformed' in locals() and
        'return_periods_prob' in locals()):

        plot_pooled_gev_fit_boxcox(
            transformed_pooled_data,
            pooled_lambda,
            regional_params_transformed,
            regional_growth_factors_transformed,
            return_periods_prob # Pass the dictionary
        )
    else:
        print("Skipping Pooled GEV fit plot: Required variables not found.")
except Exception as e:
    print(f"Error generating Pooled GEV fit plot: {e}")



# 4. Plot At-Site vs Regression Prediction Comparison
print("\n--- Checking data for At-Site vs Regression plot --- ")
try:
    # DEBUG: Check existence and type/size of required variables
    print(f"  'AtSite_df' exists: {'AtSite_df' in locals() and AtSite_df is not None}")
    if 'AtSite_df' in locals() and AtSite_df is not None: print(f"  AtSite_df shape: {AtSite_df.shape}, Index Name: {AtSite_df.index.name}")
    print(f"  'LMoments_aligned' exists: {'LMoments_aligned' in locals()}")
    if 'LMoments_aligned' in locals(): print(f"  LMoments_aligned shape: {LMoments_aligned.shape}, Index Name: {LMoments_aligned.index.name}")
    y_hat_dict_exists = 'y_hat_predictions' in locals() and isinstance(y_hat_predictions, dict)
    print(f"  'y_hat_predictions' dictionary exists: {y_hat_dict_exists}")
    if y_hat_dict_exists: print(f"  Keys in y_hat_predictions: {list(y_hat_predictions.keys())}")

    # Check if AtSite_df (created earlier) exists and other necessary data is available
    if 'AtSite_df' in locals() and AtSite_df is not None and 'LMoments_aligned' in locals() and y_hat_dict_exists:
        print("  Proceeding to create Regression_Predictions_df for plotting...")
        # Use the AtSite_df created in Section 2
        # Ensure its index is named 'Site ID' for consistency with alignment logic
        if AtSite_df.index.name != 'Site ID':
             print(" Warning: Renaming AtSite_df index to 'Site ID' for plotting.")
             AtSite_df.index.name = 'Site ID'

        # Create DataFrame for Regression Predictions using the dictionary
        # Ensure LMoments_aligned index is named 'Site ID' for alignment
        if LMoments_aligned.index.name != 'Site ID':
            print(" Warning: Renaming LMoments_aligned index to 'Site ID' for plotting alignment.")
            LMoments_aligned.index.name = 'Site ID'

        Regression_Predictions_df = pd.DataFrame({
            'Site ID': LMoments_aligned.index, # Use the index (Site ID string)
            '5_hat': y_hat_predictions['5'],
            '20_hat': y_hat_predictions['20'],
            '50_hat': y_hat_predictions['50'],
            '100_hat': y_hat_predictions['100'],
            '1000_hat': y_hat_predictions['1000']
        }).set_index('Site ID') # Set the Site ID as the index
        print(f"  Regression_Predictions_df created, shape: {Regression_Predictions_df.shape}, Index Name: {Regression_Predictions_df.index.name}")

        # Define return periods to plot
        return_periods_to_plot = ['5', '20', '50', '100', '1000']

        # Call the plotting function (which expects index intersection based on Site ID)
        print("  Calling plot_atsite_vs_regression...")
        plot_atsite_vs_regression(AtSite_df, Regression_Predictions_df, return_periods_to_plot)
        print("  Finished plot_atsite_vs_regression call.")
    else:
        print("Skipping At-Site vs Regression plot: Required data variables not available (AtSite_df, LMoments_aligned, or y_hat_predictions dict).")
except Exception as e:
    print(f"Error generating At-Site vs Regression plot: {e}")
    import traceback
    traceback.print_exc() # Add traceback for detailed error
print("--- Finished At-Site vs Regression plot block ---")

# 4. Perform Leave-One-Out Cross-Validation
print("\n--- Attempting LOOCV Execution --- ")
try:
    # Check if the prepared variables AND AtSite_df are valid
    if Xvar_unnormalized_final is not None and Yvars_for_loocv is not None and site_ids_for_loocv is not None and 'AtSite_df' in locals() and AtSite_df is not None:
        print("\n--- Starting Leave-One-Out Cross-Validation (Comparing vs At-Site GEV) --- ")
        # Ensure AtSite_df index is named 'Site ID' before passing to LOOCV
        if AtSite_df.index.name != 'Site ID':
            print(" Warning: Renaming AtSite_df index to 'Site ID' for LOOCV.")
            AtSite_df.index.name = 'Site ID'
        # Call the modified perform_loocv, passing the AtSite_df prepared earlier
        loocv_metrics, atsite_aligned_loocv, pred_aligned_loocv = perform_loocv(
            Xvar_full=Xvar_unnormalized_final,
            Yvars_dict=Yvars_for_loocv,
            site_ids=site_ids_for_loocv,
            AtSite_df_full=AtSite_df # Pass the DataFrame containing At-Site GEV estimates (indexed by Site ID)
        )
        # Optionally store/use the returned metrics and aligned dataframes
        print("\n--- LOOCV (vs At-Site) Completed --- ")
    else:
        print("\nSkipping LOOCV due to missing required data (Xvar, Yvars, Site IDs, or AtSite_df). Ensure At-Site analysis ran successfully.")
except Exception as e:
    print(f"Error during LOOCV setup or execution: {e}")

print("Script Finished.")