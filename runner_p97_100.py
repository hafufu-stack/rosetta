import sys, os, gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments'))

phases = [
    ('phase97_uncertainty', 97),
    ('phase98_holographic', 98),
    ('phase99_renormalization', 99),
    ('phase100_grand_score', 100),
]

for mod_name, num in phases:
    print(f"\n{'='*70}\n  Phase {num}\n{'='*70}")
    try:
        mod = __import__(mod_name)
        mod.main()
        gc.collect()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

import winsound, time
for _ in range(5):
    winsound.Beep(1000, 500)
    time.sleep(0.3)
