"""
Sorkin (2018) Revealed Preference Model: Simplified Implementation
==================================================================

Estimates firm values from worker mobility flows.
Assumes exogenous separation rates (δ, ρ) are known/calibrated.

3 firms (A, B, C) + non-employment (N)
"""

import numpy as np
from scipy.optimize import minimize_scalar

# =============================================================================
# DATA SETUP
# =============================================================================

# Flow matrix: M[i, j] = flows FROM state j TO state i
# States: 0=A, 1=B, 2=C, 3=N (non-employment)
# Diagonal entries are meaningless (no self-flows)

M = np.array([
    #  From A,  From B,  From C,  From N
    [    0,      50,      30,     120],   # To A
    [   40,       0,      25,      70],   # To B
    [   20,      15,       0,      30],   # To C
    [   60,      80,      50,       0],   # To N
], dtype=float)

# State labels for readability
STATES = ['A', 'B', 'C', 'N']
N_STATES = 4
FIRMS = [0, 1, 2]  # Indices for firms (excluding non-employment)
N_IDX = 3          # Index for non-employment

# Employment shares (among employed workers)
g = np.array([0.40, 0.35, 0.25])  # Firms A, B, C

# Total workers
W = 10_000   # Employed
U = 1_000    # Non-employed

# Calibrated exogenous rates (from Sorkin Table 3)
delta = 0.04  # Exogenous job destruction rate
rho = 0.03    # Exogenous reallocation rate


# =============================================================================
# STEP 1: BRADLEY-TERRY ITERATION
# =============================================================================

def bradley_terry(M, max_iter=100, tol=1e-8, verbose=False):
    """
    Estimate flow-relevant values using Bradley-Terry iteration.
    
    The model: Pr(j beats k) = exp(V_j) / (exp(V_j) + exp(V_k))
    
    Parameters
    ----------
    M : ndarray, shape (n, n)
        Flow matrix. M[j, k] = number of moves from k to j.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print iteration progress.
    
    Returns
    -------
    V_tilde : ndarray, shape (n,)
        Flow-relevant values (normalized so V_tilde[N] = 0).
    """
    n = M.shape[0]
    
    # Initialize: exp(V) = 1 for all states
    exp_V = np.ones(n)
    
    for iteration in range(max_iter):
        exp_V_old = exp_V.copy()
        
        for j in range(n):
            # Numerator: total inflows to state j
            numerator = np.sum(M[j, :])
            
            # Denominator: sum over all states k != j
            denominator = 0.0
            for k in range(n):
                if k != j:
                    # Total comparisons between j and k
                    n_comparisons = M[j, k] + M[k, j]
                    if n_comparisons > 0:
                        denominator += n_comparisons / (exp_V_old[j] + exp_V_old[k])
            
            if denominator > 0:
                exp_V[j] = numerator / denominator
        
        # Normalize: set non-employment (last state) to exp(V) = 1
        exp_V = exp_V / exp_V[N_IDX]
        
        # Check convergence
        max_change = np.max(np.abs(exp_V - exp_V_old))
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: max change = {max_change:.2e}")
        
        if max_change < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
    
    # Convert to log scale
    V_tilde = np.log(exp_V)
    
    return V_tilde


# =============================================================================
# STEP 2: ESTIMATE OFFER DISTRIBUTION
# =============================================================================

def estimate_offer_distribution(M):
    """
    Estimate offer distribution from hiring out of non-employment.
    
    Assumption: f_j ∝ M[j, N] (hires from non-employment)
    
    This is the "first approximation" that ignores rejection by non-employed.
    """
    hires_from_N = M[FIRMS, N_IDX]  # Hires from non-employment to each firm
    f = hires_from_N / np.sum(hires_from_N)
    return f


# =============================================================================
# STEP 3: RECOVER TRUE VALUES
# =============================================================================

def recover_true_values(V_tilde, f, g, delta, rho):
    """
    Unravel flow-relevant values to get true values.
    
    V^e_j = Ṽ_j - ln(f_j) + ln(g_j) + ln(1-δ) + ln(1-ρ)
    
    Parameters
    ----------
    V_tilde : ndarray
        Flow-relevant values for firms (length = n_firms)
    f : ndarray
        Offer distribution
    g : ndarray
        Employment shares
    delta, rho : float
        Exogenous separation rates
    
    Returns
    -------
    V_e : ndarray
        True firm values (relative to non-employment)
    """
    V_e = V_tilde - np.log(f) + np.log(g) + np.log(1 - delta) + np.log(1 - rho)
    return V_e


# =============================================================================
# FULL ESTIMATION
# =============================================================================

def estimate_sorkin_model(M, g, W, U, delta, rho, verbose=True):
    """
    Full estimation of the Sorkin model.
    
    Returns
    -------
    results : dict
        Dictionary containing all estimated parameters.
    """
    if verbose:
        print("=" * 60)
        print("SORKIN MODEL ESTIMATION")
        print("=" * 60)
        print(f"\nCalibrated parameters: δ = {delta:.3f}, ρ = {rho:.3f}")
        print(f"Employment: W = {W:,}, Non-employment: U = {U:,}")
    
    # Step 1: Bradley-Terry
    if verbose:
        print("\n--- Step 1: Bradley-Terry Iteration ---")
    V_tilde_full = bradley_terry(M, verbose=verbose)
    V_tilde_firms = V_tilde_full[FIRMS]
    
    if verbose:
        print("\nFlow-relevant values (Ṽ):")
        for i, s in enumerate(STATES):
            print(f"  {s}: {V_tilde_full[i]:.4f}")
    
    # Step 2: Offer distribution
    if verbose:
        print("\n--- Step 2: Offer Distribution ---")
    f = estimate_offer_distribution(M)
    
    if verbose:
        print("Offer probabilities (f):")
        for i, firm in enumerate(['A', 'B', 'C']):
            print(f"  {firm}: {f[i]:.4f}")
    
    # Step 3: Find λ₁
    # First need preliminary V_e to compute acceptance probabilities
    # Use V_tilde as initial approximation
    V_e_prelim = recover_true_values(V_tilde_firms, f, g, delta, rho)
    
    if verbose:
        print("\n--- Step 3: Find λ₁ ---")
    lambda1, obs_ee_rate = find_lambda1(M, V_e_prelim, f, g, W, delta, rho)
    
    if verbose:
        print(f"Observed EE rate: {obs_ee_rate:.4f}")
        print(f"Estimated λ₁: {lambda1:.4f}")
        
        # Verify
        model_rate = model_ee_rate(lambda1, V_e_prelim, f, g, delta, rho)
        print(f"Model-implied EE rate: {model_rate:.4f}")
    
    # Step 4: Final true values
    if verbose:
        print("\n--- Step 4: True Values ---")
    V_e = recover_true_values(V_tilde_firms, f, g, delta, rho)
    
    if verbose:
        print("True firm values (V^e, relative to non-employment):")
        for i, firm in enumerate(['A', 'B', 'C']):
            print(f"  {firm}: {V_e[i]:.4f}")
    
    # Package results
    results = {
        'V_tilde': V_tilde_full,
        'V_tilde_firms': V_tilde_firms,
        'f': f,
        'lambda1': lambda1,
        'V_e': V_e,
        'observed_ee_rate': obs_ee_rate,
    }
    
    return results


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_diagnostics(M, results, g, delta, rho):
    """
    Print model fit diagnostics.
    """
    V_tilde = results['V_tilde']
    
    print("\n" + "=" * 60)
    print("MODEL FIT DIAGNOSTICS")
    print("=" * 60)
    
    print("\n--- Relative Flows vs Model Predictions ---")
    print("(For EE flows: does M[j,k]/M[k,j] ≈ exp(Ṽ_j)/exp(Ṽ_k)?)\n")
    
    for j in FIRMS:
        for k in FIRMS:
            if j < k:  # Only print each pair once
                if M[k, j] > 0 and M[j, k] > 0:
                    observed_ratio = M[j, k] / M[k, j]
                    model_ratio = np.exp(V_tilde[j]) / np.exp(V_tilde[k])
                    
                    print(f"  {STATES[j]} vs {STATES[k]}:")
                    print(f"    M[{STATES[j]},{STATES[k]}]/M[{STATES[k]},{STATES[j]}] = "
                          f"{M[j,k]:.0f}/{M[k,j]:.0f} = {observed_ratio:.3f}")
                    print(f"    exp(Ṽ_{STATES[j]})/exp(Ṽ_{STATES[k]}) = {model_ratio:.3f}")
                    print()
    
    # Choice probabilities
    print("--- Implied Acceptance Probabilities ---")
    print("Pr(accept offer from k | currently at j)\n")
    
    V_e = results['V_e']
    exp_V = np.exp(V_e)
    
    print("       ", end="")
    for k in ['A', 'B', 'C']:
        print(f"  Offer={k}", end="")
    print()
    
    for j, j_label in enumerate(['A', 'B', 'C']):
        print(f"At {j_label}:  ", end="")
        for k in range(3):
            if k == j:
                print(f"    ---  ", end="")
            else:
                pr = exp_V[k] / (exp_V[k] + exp_V[j])
                print(f"   {pr:.3f} ", end="")
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Print input data
    print("INPUT DATA")
    print("=" * 60)
    print("\nFlow Matrix (rows=destination, cols=origin):")
    print("       From A  From B  From C  From N")
    for i, row_label in enumerate(STATES):
        print(f"To {row_label}:", end="")
        for j in range(4):
            print(f"  {M[i,j]:6.0f}", end="")
        print()
    
    print(f"\nEmployment shares: g_A={g[0]}, g_B={g[1]}, g_C={g[2]}")
    print(f"Workers: W={W:,}, U={U:,}")
    
    # Run estimation
    results = estimate_sorkin_model(M, g, W, U, delta, rho, verbose=True)
    
    # Diagnostics
    print_diagnostics(M, results, g, delta, rho)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nEstimated Parameters:")
    print(f"  Offer arrival rate (λ₁): {results['lambda1']:.4f}")
    print(f"\nOffer Distribution:")
    for i, firm in enumerate(['A', 'B', 'C']):
        print(f"  f_{firm} = {results['f'][i]:.4f}")
    print(f"\nFirm Values (V^e, relative to non-employment V^n=0):")
    for i, firm in enumerate(['A', 'B', 'C']):
        print(f"  V^e_{firm} = {results['V_e'][i]:.4f}")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print("""
The flow-relevant values (Ṽ) show Firm A appears most attractive
in raw mobility patterns. However, after adjusting for:
  - Offer rates (Firm A makes more offers: f_A > f_B > f_C)
  - Firm size (Firm A is larger)
  
The true values (V^e) are much closer together. This illustrates
Sorkin's key point: raw mobility flows conflate true firm value
with recruiting intensity and firm size.
""")