import time
from keystone.pipeline import evaluate_stacking
t0=time.time()
print(">>> RUN 1: franka, n=4, sims=2000, seed=0 (real search)", flush=True)
r1=evaluate_stacking(n=4, dx=1.0/12.0, sims=2000, seed=0, executor="franka",
    record=True, out_dir="out/pipeline", settle_duration=2.0, verbose=True)
print("RUN1 done %.0fs"%(time.time()-t0), flush=True)
t1=time.time()
print("\n>>> RUN 2: franka, archived 26/24 control, cw transient prop", flush=True)
r2=evaluate_stacking(n=4, dx=1.0/24.0, sims=0, seed=0,
    sequence=[(0,-2),(1,-14),(2,-4),(1,14)], prop_steps=[1],
    executor="franka", record=True, out_dir="out/pipeline",
    tag="_ctrl26_24", settle_duration=2.0, verbose=True)
print("RUN2 done %.0fs"%(time.time()-t1), flush=True)
print("\nTOTAL %.0fs"%(time.time()-t0), flush=True)
