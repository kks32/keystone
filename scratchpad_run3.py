import time
from keystone.pipeline import evaluate_stacking
t0=time.time()
print(">>> RUN 3: franka, n=6, sims=4000, seed=0 (1.4167 discovery)", flush=True)
r3=evaluate_stacking(n=6, dx=1.0/12.0, sims=4000, seed=0, executor="franka",
    record=True, out_dir="out/pipeline", settle_duration=2.0, verbose=True)
print("RUN3 done %.0fs"%(time.time()-t0), flush=True)
