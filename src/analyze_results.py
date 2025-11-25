
# Simple analyzer that plots convergence and writes a short summary.
import os, json, matplotlib.pyplot as plt
from glob import glob
def cummin(arr):
    out=[]; m=float('inf')
    for x in arr:
        if x<m: m=x
        out.append(m)
    return out
bo = sorted(glob('results/run_bo_*.json'))
rs = sorted(glob('results/run_rs_*.json'))
bo_vals=[json.load(open(p))['best_val_loss'] for p in bo] if bo else []
rs_vals=[json.load(open(p))['best_val_loss'] for p in rs] if rs else []
bo_c=cummin(bo_vals); rs_c=cummin(rs_vals)
plt.figure()
if bo_c: plt.plot(range(1,len(bo_c)+1), bo_c)
if rs_c: plt.plot(range(1,len(rs_c)+1), rs_c)
plt.xlabel('Iteration'); plt.ylabel('Best validation loss so far'); plt.title('Convergence: BO vs Random')
os.makedirs('results', exist_ok=True); plt.savefig('results/convergence.png')
with open('results/analysis_summary.json','w') as f: json.dump({'bo_last': bo_c[-1] if bo_c else None, 'rs_last': rs_c[-1] if rs_c else None}, f, indent=2)
print('Analysis saved to results/analysis_summary.json and convergence.png')
