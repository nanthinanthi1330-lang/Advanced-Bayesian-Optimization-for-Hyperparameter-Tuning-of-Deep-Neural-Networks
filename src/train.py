
# Minimal training wrapper for CIFAR-10. Writes JSON with best_val_loss and history.
import argparse, json, os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T, torchvision.datasets as datasets
from baseline_model import SimpleCNN

def get_dataloaders(batch_size):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    val = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size, shuffle=False)

def run_training(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(cfg['batch_size'])
    model = SimpleCNN(num_classes=10, base_filters=cfg['base_filters'], dropout=cfg['dropout']).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    best_val = float('inf')
    history = {'val_loss':[]}
    for e in range(cfg['epochs']):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()
        # eval
        model.eval()
        running=0.0
        total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out,y)
                running += loss.item()*x.size(0)
                total += x.size(0)
        val_loss = running/total
        history['val_loss'].append(val_loss)
        if val_loss < best_val: best_val = val_loss
    return best_val, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--out', type=str, default='results/run.json')
    args = parser.parse_args()
    cfg = {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'dropout': args.dropout, 'base_filters': args.base_filters}
    best_val, history = run_training(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w') as f:
        json.dump({'best_val_loss': best_val, 'history': history, 'config': cfg}, f, indent=2)
    print('Saved', args.out)
