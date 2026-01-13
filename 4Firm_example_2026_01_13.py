"""
Sorkin's Revealed Preference Algorithm for Estimating Firm Values
=================================================================

This implements the rank-aggregation method from Section 2.3 of Sorkin (2018),
"Ranking Firms Using Revealed Preference" (QJE).

The algorithm estimates firm values V^EE from worker flows between firms,
using the insight that workers reveal preferences through their mobility choices.
"""

import numpy as np

def estimate_firm_values(transition_matrix, firm_names=None):
    """
    Estimate firm values using Sorkin's revealed preference algorithm.
    
    Parameters
    ----------
    transition_matrix : ndarray, shape (n_firms, n_firms)
        Entry (i, j) = number of workers moving FROM firm i TO firm j.
        Diagonal entries should be 0 (or will be ignored).
    
    firm_names : list of str, optional
        Names for each firm (for display purposes)
    
    Returns
    -------
    dict with keys:
        'V_EE': log firm values (normalized so first firm = 0)
        'exp_V_EE': exp(V_EE), the eigenvector
        'ranking': firms sorted from best to worst
        'eigenvalues': all eigenvalues of the normalized flow matrix
        'spectral_gap': difference between 1st and 2nd largest eigenvalues
    """
    n_firms = transition_matrix.shape[0]
    
    if firm_names is None:
        firm_names = [f"Firm {i+1}" for i in range(n_firms)]
    
    # Step 1: Construct M^o (flows TO row FROM column)
    # M_kj = flows into k from j = transpose of transition matrix
    M = transition_matrix.T
    
    # Step 2: Compute exits from each firm
    # exits[k] = sum_j M_jk = total workers leaving firm k
    # This is the column sum of M (or equivalently, row sum of transition_matrix)
    exits = M.sum(axis=0)
    
    # Check for firms with no exits (would cause division by zero)
    if np.any(exits == 0):
        zero_exit_firms = [firm_names[i] for i in np.where(exits == 0)[0]]
        raise ValueError(f"Firms with no exits (value undefined): {zero_exit_firms}")
    
    # Step 3: Construct normalized flow matrix P
    # P_kj = M_kj / exits[k]
    # Interpretation: probability-weighted value of where workers come FROM
    P = np.zeros_like(M, dtype=float)
    for k in range(n_firms):
        P[k, :] = M[k, :] / exits[k]
    
    # Step 4: Find eigenvector for eigenvalue 1
    # This solves: P @ exp(V) = exp(V)
    eigenvalues, eigenvectors = np.linalg.eig(P)
    
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    
    # Get corresponding eigenvector (take real part)
    exp_V = eigenvectors[:, idx].real
    
    # Normalize so first firm has exp(V) = 1, i.e., V = 0
    exp_V = exp_V / exp_V[0]
    
    # Step 5: Take logs to get V^EE
    V_EE = np.log(exp_V)
    
    # Compute spectral gap (measure of estimation precision)
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    spectral_gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    
    # Create ranking
    ranking_idx = np.argsort(V_EE)[::-1]
    ranking = [firm_names[i] for i in ranking_idx]
    
    return {
        'V_EE': V_EE,
        'exp_V_EE': exp_V,
        'ranking': ranking,
        'eigenvalues': eigenvalues,
        'spectral_gap': spectral_gap,
        'normalized_flow_matrix': P,
        'exits': exits
    }


def print_results(results, firm_names):
    """Pretty-print the estimation results."""
    print("=" * 60)
    print("SORKIN ALGORITHM RESULTS")
    print("=" * 60)
    
    print("\nFirm Values (V^EE):")
    print("-" * 30)
    for i, name in enumerate(firm_names):
        print(f"  {name}: {results['V_EE'][i]:+.4f}")
    
    print(f"\nRanking (best to worst): {' > '.join(results['ranking'])}")
    
    print(f"\nSpectral gap: {results['spectral_gap']:.4f}")
    print("  (larger = more precise estimation)")
    
    print(f"\nEigenvalues: {np.round(results['eigenvalues'].real, 4)}")


# =============================================================================
# EXAMPLE: 4 FIRMS
# =============================================================================

if __name__ == "__main__":
    
    # Define firm names
    firms = ["A", "B", "C", "D"]
    
    # Transition matrix: entry (i,j) = workers moving FROM firm i TO firm j
    #
    #         To:    A    B    C    D
    # From:
    transitions = np.array([
        #  A    B    C    D
        [  0,  40,  20,   5],  # From A
        [ 50,   0,  30,  10],  # From B
        [ 30,  40,   0,  25],  # From C
        [ 10,  15,  35,   0],  # From D
    ])
    
    print("Input: Transition Matrix")
    print("(rows = origin, columns = destination)")
    print()
    print("       ", "  ".join(f"{f:>4}" for f in firms))
    for i, row in enumerate(transitions):
        print(f"  {firms[i]}:  ", "  ".join(f"{x:4d}" for x in row))
    print()
    
    # Estimate firm values
    results = estimate_firm_values(transitions, firm_names=firms)
    
    # Print results
    print_results(results, firms)
    
    # Show intermediate calculations
    print("\n" + "=" * 60)
    print("INTERMEDIATE CALCULATIONS")
    print("=" * 60)
    
    print("\nFlow matrix M^o (rows = destination, cols = origin):")
    M = transitions.T
    print("       ", "  ".join(f"{f:>6}" for f in firms))
    for i, row in enumerate(M):
        print(f"  {firms[i]}:  ", "  ".join(f"{x:6d}" for x in row))
    
    print(f"\nTotal exits from each firm: {dict(zip(firms, results['exits'].astype(int)))}")
    
    print("\nNormalized flow matrix P = S^{-1} M^o:")
    print("       ", "  ".join(f"{f:>6}" for f in firms))
    for i, row in enumerate(results['normalized_flow_matrix']):
        print(f"  {firms[i]}:  ", "  ".join(f"{x:6.3f}" for x in row))
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
The algorithm finds that A > B > C > D in terms of worker-revealed value.

Intuition: 
- Firm A attracts more workers from other firms relative to how many it loses
- Firm D loses more workers to other firms than it attracts
- The recursive structure means a firm is valuable if workers come from 
  OTHER valuable firms (PageRank logic)

Key insight: Even if direct A<->D flows are sparse, the algorithm infers
relative values through chains like A<->B<->C<->D.
""")