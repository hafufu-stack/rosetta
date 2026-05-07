"""Runner for Chapter XXVI: The Holographic Universe (P101-105)"""
import sys, os, gc

EXP2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments2')
sys.path.insert(0, EXP2_DIR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

phases = [
    ('phase101_holographic_decoder', 101),
    ('phase102_golden_ast', 102),
    ('phase103_annealing', 103),
    ('phase104_dark_matter', 104),
    ('phase105_cosmic_web', 105),
]

if __name__ == '__main__':
    results = {}
    for mod_name, num in phases:
        print(f"\n{'='*70}")
        print(f"  Phase {num}: {mod_name}")
        print(f"{'='*70}\n")
        try:
            mod = __import__(mod_name)
            r = mod.main()
            results[num] = r
            gc.collect()
            print(f"\n  Phase {num} OK!")
        except Exception as e:
            print(f"\n  Phase {num} FAILED: {e}")
            import traceback; traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("  Chapter XXVI Complete!")
    print(f"{'='*70}")
    for num, r in results.items():
        t = r.get('title', '')
        l = r.get('law', '')[:100]
        print(f"  P{num}: {t}")
        print(f"    -> {l}")
    
    import winsound, time
    for _ in range(5):
        winsound.Beep(1000, 500)
        time.sleep(0.3)
