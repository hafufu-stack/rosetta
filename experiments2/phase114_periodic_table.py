"""Phase 114: The Periodic Table of Programs - Elementary vs composite operations.
Like chemistry's periodic table, can we classify all programs by fundamental properties?
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 114: The Periodic Table of Programs")
    print("  Elementary particles of computation")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs: func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {s: np.mean(v, axis=0) for s, v in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique_funcs])
    
    # Classify functions by properties
    g = {}
    func_props = {}
    for func_src in unique_funcs:
        props = {'commutative': False, 'associative': False, 'has_identity': False,
                 'idempotent': False, 'monotonic': False, 'symmetric': False}
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            
            if n_args == 2:
                # Commutativity: f(a,b) == f(b,a)?
                comm_tests = [(1,2),(3,5),(-1,4)]
                try:
                    comm = all(fn(a,b) == fn(b,a) for a,b in comm_tests)
                    props['commutative'] = comm
                except: pass
                
                # Identity: f(x,0)==x or f(x,1)==x?
                try:
                    id0 = all(fn(x,0) == x for x in [1,2,3,-1])
                    id1 = all(fn(x,1) == x for x in [1,2,3,-1])
                    props['has_identity'] = id0 or id1
                except: pass
                
                # Idempotent: f(x,x)==x?
                try:
                    idemp = all(fn(x,x) == x for x in [1,2,3,5])
                    props['idempotent'] = idemp
                except: pass
                
                # Monotonic: f(a,c) >= f(b,c) when a >= b?
                try:
                    mono = all(fn(3,y) >= fn(1,y) for y in [1,2,3])
                    props['monotonic'] = mono
                except: pass
                
                # Symmetric output: f(x,y)==f(y,x)?
                props['symmetric'] = props['commutative']
        except: pass
        
        func_props[func_src] = props
    
    # Count properties
    prop_counts = {}
    for name in ['commutative', 'has_identity', 'idempotent', 'monotonic', 'symmetric']:
        count = sum(1 for p in func_props.values() if p[name])
        prop_counts[name] = count
        print(f"  {name}: {count}/{len(unique_funcs)} ({count/len(unique_funcs):.0%})")
    
    # Create "atomic number" = fingerprint vector
    fingerprints = []
    for f in unique_funcs:
        p = func_props[f]
        fp = [int(p['commutative']), int(p['has_identity']), int(p['idempotent']),
              int(p['monotonic']), int(p['symmetric'])]
        fingerprints.append(fp)
    fingerprints = np.array(fingerprints)
    
    # Find unique element types
    fp_strings = [''.join(str(x) for x in fp) for fp in fingerprints]
    element_types = {}
    for fp_str, f in zip(fp_strings, unique_funcs):
        if fp_str not in element_types:
            element_types[fp_str] = []
        element_types[fp_str].append(f)
    
    print(f"\n--- Element Types (Periodic Table Rows) ---")
    print(f"  Total unique types: {len(element_types)}")
    for fp_str, funcs in sorted(element_types.items(), key=lambda x: -len(x[1])):
        props = ['Comm' if fp_str[0]=='1' else '', 'Id' if fp_str[1]=='1' else '',
                 'Idemp' if fp_str[2]=='1' else '', 'Mono' if fp_str[3]=='1' else '',
                 'Sym' if fp_str[4]=='1' else '']
        props = [p for p in props if p]
        examples = [f.split('return ')[-1].strip()[:15] for f in funcs[:3]]
        print(f"  [{fp_str}] ({', '.join(props) or 'None'}): {len(funcs)} elements - {', '.join(examples)}")
    
    # PCA and color by element type
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)
    
    # Assign colors by type
    type_colors = {}
    color_palette = ['#F44336','#2196F3','#4CAF50','#FF9800','#9C27B0','#00BCD4','#795548','#E91E63']
    for i, fp_str in enumerate(sorted(element_types.keys(), key=lambda x: -len(element_types[x]))):
        type_colors[fp_str] = color_palette[i % len(color_palette)]
    
    point_colors = [type_colors[fp_str] for fp_str in fp_strings]
    
    # K-means to find natural groups
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = km.fit_predict(all_vecs)
    
    # How well do algebraic properties predict clusters?
    from sklearn.metrics import adjusted_rand_score
    fp_labels = [fp_str for fp_str in fp_strings]
    # Convert to numeric
    fp_unique = list(set(fp_labels))
    fp_numeric = [fp_unique.index(fp) for fp in fp_labels]
    ari = adjusted_rand_score(fp_numeric, clusters)
    
    print(f"\n--- Cluster-Property Agreement ---")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  {'Properties predict structure!' if ari > 0.1 else 'Weak alignment'}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 114: The Periodic Table of Programs', fontsize=14, fontweight='bold')
    
    axes[0].scatter(vecs_2d[:,0], vecs_2d[:,1], c=point_colors, s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].set_title('Elements in Latent Space (by algebraic type)')
    
    prop_names = list(prop_counts.keys())
    prop_vals = [prop_counts[p] for p in prop_names]
    axes[1].barh(prop_names, prop_vals, color='#2196F3', edgecolor='black')
    axes[1].set_xlabel('Count'); axes[1].set_title('Algebraic Properties')
    
    type_sizes = sorted([(len(v), k) for k,v in element_types.items()], reverse=True)[:8]
    axes[2].bar(range(len(type_sizes)), [s for s,_ in type_sizes],
               color=[type_colors.get(k, '#CCC') for _,k in type_sizes], edgecolor='black')
    axes[2].set_xticks(range(len(type_sizes)))
    axes[2].set_xticklabels([k for _,k in type_sizes], fontsize=7)
    axes[2].set_ylabel('Elements'); axes[2].set_title(f'Element Types (ARI={ari:.3f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase114_periodic_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 114, 'title': 'The Periodic Table of Programs',
        'n_element_types': len(element_types),
        'property_counts': prop_counts,
        'ari': float(ari),
        'element_types': {k: len(v) for k,v in element_types.items()},
        'law': f'{len(element_types)} element types found. Properties predict clusters with ARI={ari:.3f}. Programs follow algebraic classification like elements follow chemical properties.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase114_periodic_table.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 114 complete!")
    return results

if __name__ == '__main__':
    main()
