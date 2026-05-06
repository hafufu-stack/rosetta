"""Runner for Chapter XXIV: Software Chemistry & The Singularity (P87-90)"""
import sys, os, gc

EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
sys.path.insert(0, EXPERIMENT_DIR)

def run_phase(module_name, phase_num):
    print(f"\n{'='*70}")
    print(f"  Starting Phase {phase_num}: {module_name}")
    print(f"{'='*70}\n")
    mod = __import__(module_name)
    result = mod.main()
    gc.collect()
    return result

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    phases = [
        ('phase87_molecular_synthesis', 87),
        ('phase88_thermodynamic_debugging', 88),
        ('phase89_singularity', 89),
        ('phase90_tensor', 90),
    ]
    
    results = {}
    for module_name, phase_num in phases:
        try:
            result = run_phase(module_name, phase_num)
            results[phase_num] = result
            print(f"\n  Phase {phase_num} completed successfully!")
        except Exception as e:
            print(f"\n  Phase {phase_num} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Beep notification
    print("\n" + "="*70)
    print("  Chapter XXIV Complete!")
    print("="*70)
    
    try:
        import winsound, time
        for _ in range(5):
            winsound.Beep(1000, 500)
            time.sleep(0.3)
    except:
        pass
    
    # Summary
    print("\n--- Chapter XXIV Summary ---")
    for phase_num, result in results.items():
        title = result.get('title', 'Unknown')
        law = result.get('law', '')
        print(f"  P{phase_num}: {title}")
        print(f"    -> {law[:100]}")
