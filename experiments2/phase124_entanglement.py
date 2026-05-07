"""Phase 124: Entanglement Entropy - von Neumann entropy of the program bipartite system.
Opus original: If AST and BC are 'entangled', the reduced density matrix
should have non-zero von Neumann entropy. This measures how much
information about one modality is encoded in the other.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(rho * log(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def main():
    print("=" * 60)
    print("Phase 124: Entanglement Entropy")
    print("  von Neumann entropy of the AST-BC bipartite system")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    nl_vectors = latents['nl']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast, func_bc, func_nl = {}, {}, {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []; func_bc[src] = []; func_nl[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
        func_nl[src].append(nl_vectors[i])
    
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    bc_m = np.array([np.mean(func_bc[f], axis=0) for f in unique_funcs])
    nl_m = np.array([np.mean(func_nl[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # 1. Construct "density matrix" for each modality
    # rho = (1/n) * sum_i |v_i><v_i| (normalized outer products)
    def density_matrix(vectors):
        """Compute density matrix from vectors (project to top-k dims first)."""
        k = min(32, vectors.shape[1])
        pca = PCA(n_components=k).fit(vectors)
        V = pca.transform(vectors)
        # Normalize each vector
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        V = V / norms
        rho = V.T @ V / n
        # Ensure rho is a valid density matrix (trace = 1)
        rho /= np.trace(rho)
        return rho
    
    rho_ast = density_matrix(ast_m)
    rho_bc = density_matrix(bc_m)
    rho_nl = density_matrix(nl_m)
    
    S_ast = von_neumann_entropy(rho_ast)
    S_bc = von_neumann_entropy(rho_bc)
    S_nl = von_neumann_entropy(rho_nl)
    
    print(f"--- von Neumann Entropy (bits) ---")
    print(f"  S(AST) = {S_ast:.4f}")
    print(f"  S(BC)  = {S_bc:.4f}")
    print(f"  S(NL)  = {S_nl:.4f}")
    
    # 2. Joint system entropy
    # Combine AST+BC into a joint system
    joint_ast_bc = np.hstack([ast_m, bc_m])
    rho_joint_ab = density_matrix(joint_ast_bc)
    S_joint_ab = von_neumann_entropy(rho_joint_ab)
    
    joint_ast_nl = np.hstack([ast_m, nl_m])
    rho_joint_an = density_matrix(joint_ast_nl)
    S_joint_an = von_neumann_entropy(rho_joint_an)
    
    print(f"\n--- Joint Entropies ---")
    print(f"  S(AST,BC) = {S_joint_ab:.4f}")
    print(f"  S(AST,NL) = {S_joint_an:.4f}")
    
    # 3. Mutual information = S(A) + S(B) - S(A,B)
    MI_ast_bc = S_ast + S_bc - S_joint_ab
    MI_ast_nl = S_ast + S_nl - S_joint_an
    
    print(f"\n--- Mutual Information ---")
    print(f"  I(AST;BC) = {MI_ast_bc:.4f} bits")
    print(f"  I(AST;NL) = {MI_ast_nl:.4f} bits")
    
    # 4. Entanglement entropy: S(A|B) = S(A,B) - S(B)
    # If S(A|B) < S(A), the systems are entangled
    cond_ast_given_bc = S_joint_ab - S_bc
    cond_ast_given_nl = S_joint_an - S_nl
    
    print(f"\n--- Conditional Entropy ---")
    print(f"  S(AST|BC) = {cond_ast_given_bc:.4f} (S(AST)={S_ast:.4f})")
    print(f"  S(AST|NL) = {cond_ast_given_nl:.4f} (S(AST)={S_ast:.4f})")
    print(f"  Entanglement = {'YES' if cond_ast_given_bc < S_ast * 0.5 else 'Partial'}")
    
    # 5. Eigenspectrum analysis of density matrices
    eigs_ast = np.sort(np.linalg.eigvalsh(rho_ast))[::-1]
    eigs_bc = np.sort(np.linalg.eigvalsh(rho_bc))[::-1]
    eigs_nl = np.sort(np.linalg.eigvalsh(rho_nl))[::-1]
    
    # Purity = Tr(rho^2)
    purity_ast = float(np.trace(rho_ast @ rho_ast))
    purity_bc = float(np.trace(rho_bc @ rho_bc))
    purity_nl = float(np.trace(rho_nl @ rho_nl))
    
    print(f"\n--- Purity (1=pure, 0=maximally mixed) ---")
    print(f"  P(AST) = {purity_ast:.4f}")
    print(f"  P(BC)  = {purity_bc:.4f}")
    print(f"  P(NL)  = {purity_nl:.4f}")
    
    # 6. Schmidt decomposition of joint AST-BC
    # The Schmidt coefficients tell us the entanglement structure
    joint = np.hstack([ast_m, bc_m])
    U, sigma, Vt = np.linalg.svd(joint, full_matrices=False)
    sigma_normalized = sigma / np.sum(sigma)
    schmidt_entropy = -np.sum(sigma_normalized * np.log2(sigma_normalized + 1e-15))
    
    print(f"\n--- Schmidt Decomposition ---")
    print(f"  Schmidt entropy: {schmidt_entropy:.4f} bits")
    print(f"  Top 5 Schmidt coefficients: {', '.join(f'{s:.4f}' for s in sigma_normalized[:5])}")
    print(f"  Effective Schmidt rank: {float(np.sum(sigma_normalized > 0.01))}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 124: Entanglement Entropy', fontsize=14, fontweight='bold')
    
    # Panel 1: Entropy diagram
    labels = ['S(AST)', 'S(BC)', 'S(NL)', 'S(AST,BC)', 'I(AST;BC)']
    values = [S_ast, S_bc, S_nl, S_joint_ab, MI_ast_bc]
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#9C27B0', '#FF9800']
    axes[0].bar(labels, values, color=colors, edgecolor='black')
    axes[0].set_ylabel('Entropy (bits)'); axes[0].set_title('Information quantities')
    axes[0].tick_params(axis='x', rotation=30)
    
    # Panel 2: Eigenvalue spectrum of density matrices
    axes[1].semilogy(eigs_ast[:20], 'o-', color='#E91E63', label='AST', markersize=4)
    axes[1].semilogy(eigs_bc[:20], 's-', color='#2196F3', label='BC', markersize=4)
    axes[1].semilogy(eigs_nl[:20], '^-', color='#4CAF50', label='NL', markersize=4)
    axes[1].set_xlabel('Eigenvalue index'); axes[1].set_ylabel('Eigenvalue (log)')
    axes[1].set_title('Density matrix spectra'); axes[1].legend()
    
    # Panel 3: Schmidt coefficients
    axes[2].bar(range(min(20, len(sigma_normalized))), sigma_normalized[:20],
               color='#FF5722', edgecolor='black')
    axes[2].set_xlabel('Schmidt index'); axes[2].set_ylabel('Coefficient')
    axes[2].set_title(f'Schmidt decomposition (S={schmidt_entropy:.2f} bits)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase124_entanglement.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 124, 'title': 'Entanglement Entropy',
        'entropy': {'AST': float(S_ast), 'BC': float(S_bc), 'NL': float(S_nl)},
        'joint_entropy': {'AST_BC': float(S_joint_ab), 'AST_NL': float(S_joint_an)},
        'mutual_information': {'AST_BC': float(MI_ast_bc), 'AST_NL': float(MI_ast_nl)},
        'purity': {'AST': purity_ast, 'BC': purity_bc, 'NL': purity_nl},
        'schmidt_entropy': float(schmidt_entropy),
        'effective_schmidt_rank': float(np.sum(sigma_normalized > 0.01)),
        'law': f'S(AST)={S_ast:.3f}, S(BC)={S_bc:.3f}. I(AST;BC)={MI_ast_bc:.3f} bits. Schmidt entropy={schmidt_entropy:.3f} bits. Purity: AST={purity_ast:.4f}, BC={purity_bc:.4f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase124_entanglement.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 124 complete!")
    return results

if __name__ == '__main__':
    main()
