
# Random search baseline
import random, json, os, subprocess, time
def call_train(cfg, out_path):
    cmd = ['python', 'src/train.py', '--epochs', str(cfg['epochs']), '--batch-size', str(cfg['batch_size']), '--lr', str(cfg['lr']), '--dropout', str(cfg['dropout']), '--base-filters', str(cfg['base_filters']), '--out', out_path]
    return subprocess.run(cmd).returncode

def run_random(n_trials=20):
    os.makedirs('results', exist_ok=True)
    for i in range(n_trials):
        cfg = {'lr': 10**random.uniform(-5,-2), 'batch_size': random.choice([32,64,128,256]), 'dropout': random.uniform(0.0,0.6), 'base_filters': random.choice([16,32,48,64]), 'epochs': random.randint(3,12)}
        ts = int(time.time())
        out = f'results/run_rs_{ts}.json'
        rc = call_train(cfg, out)
        if rc != 0: continue
        with open(out,'r') as f:
            r = json.load(f)
        with open('results/random_summary.jsonl','a') as s:
            s.write(json.dumps({'cfg':cfg,'val':r.get('best_val_loss')})+'\n')
    print('Random search complete.')
if __name__ == '__main__':
    run_random()
