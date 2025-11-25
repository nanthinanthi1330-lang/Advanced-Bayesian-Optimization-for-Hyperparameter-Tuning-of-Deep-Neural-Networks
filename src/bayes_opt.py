
# Bayesian Optimization wrapper using scikit-optimize gp_minimize.
import json, os, time, subprocess
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

space = [
    Real(1e-5, 1e-2, name='lr', prior='log-uniform'),
    Integer(32, 256, name='batch_size'),
    Real(0.0, 0.6, name='dropout'),
    Integer(16, 64, name='base_filters'),
    Integer(3, 12, name='epochs')
]

def call_train(cfg, out_path):
    cmd = ['python', 'src/train.py', '--epochs', str(cfg['epochs']), '--batch-size', str(cfg['batch_size']), '--lr', str(cfg['lr']), '--dropout', str(cfg['dropout']), '--base-filters', str(cfg['base_filters']), '--out', out_path]
    return subprocess.run(cmd).returncode

@use_named_args(space)
def objective(**params):
    cfg = {'lr': params['lr'], 'batch_size': int(params['batch_size']), 'dropout': float(params['dropout']), 'base_filters': int(params['base_filters']), 'epochs': int(params['epochs'])}
    ts = int(time.time())
    out = f'results/run_bo_{ts}.json'
    os.makedirs('results', exist_ok=True)
    rc = call_train(cfg, out)
    if rc != 0:
        return 1.0
    with open(out,'r') as f:
        r = json.load(f)
    val = r.get('best_val_loss', 1.0)
    with open('results/bo_summary.jsonl','a') as s:
        s.write(json.dumps({'cfg':cfg,'val':val})+'\n')
    return float(val)

def run_bo(n_calls=20):
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
    best_cfg = {dim.name: val for dim,val in zip(res.space.dimensions, res.x)}
    os.makedirs('results', exist_ok=True)
    with open('results/final_bo_config.json','w') as f:
        json.dump({'best_cfg':best_cfg,'best_val':res.fun}, f, indent=2)
    print('Saved results/final_bo_config.json')

if __name__ == '__main__':
    run_bo()
