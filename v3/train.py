import torch
import copy
import mlflow                          # << NOVO
from datetime import datetime as dt
from pathlib import Path
from typing import Callable
import torch.nn as nn

try: from . import helper
except ImportError:
    import helper

__all__ = ['train', 'merge']

@helper.timer
def train(
    filepaths: helper.DirDataset | str | Path,
    /, *, epochs: int = 3,
    batch_size: int = 64,
    ohkeep: float = 0.5,
    ohwarmup: int = 1,
    ohalpha: float = 0.5,
    clip: float = 1.0,
    freeze: int = 0,
    limit: int | None = None,
    lr: float = 5e-4,
    wd: float = 1e-5,
    check: Callable|None=None,
    run_name: str | None = None,       # << NOVO: nome opcional para o run
):
    # Train for two-class detection: 0=fake, 1=real.
    # Supports optional Online Hard Example Mining (OHEM).

    filepaths = helper.DirDataset(filepaths, 'train',
                    limit=limit, shuffle=True,
                    transform=helper.transform(train=True),
                    )

    ohalpha = float(ohalpha)
    if not (0.0 <= ohalpha <= 1.0):
        raise ValueError('ohalpha must be between 0 and 1')

    device = helper.best_device()

    train_loader = torch.utils.data.DataLoader(
        filepaths, batch_size=batch_size, shuffle=True,
        num_workers=4, persistent_workers=True)

    treinable = helper.freeze(freeze)
    opt = torch.optim.AdamW(treinable, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( opt,
        mode='max', factor=0.7, patience=0,
        threshold=0.005, min_lr=5e-6, cooldown=0, )

    helper.model.to(device)
    centropy = torch.nn.CrossEntropyLoss(reduction='none')

    # ------------------------------------------------------------------ NOVO
    with mlflow.start_run(run_name=run_name):

        # Registra todos os hiperparâmetros de uma vez
        mlflow.log_params({
            "epochs":     epochs,
            "batch_size": batch_size,
            "lr":         lr,
            "wd":         wd,
            "ohkeep":     ohkeep,
            "ohalpha":    ohalpha,
            "ohwarmup":   ohwarmup,
            "freeze":     freeze,
            "clip":       clip,
            "limit":      limit,
            "device":     str(device),
        })
    # ------------------------------------------------------------------ /NOVO

        print('Starting training...')
        for epoch in range(epochs):
            if callable(check): check()

            helper.model.train()
            now = dt.now()
            total_loss    = torch.tensor(0.0, device=device)
            total         = 0
            correct_total = torch.tensor(0, device=device)
            selected_total = 0

            for batch in train_loader:
                xb, yb = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.long().to(device, non_blocking=True)

                logits = helper.expand(helper.model(xb))
                losses = centropy(logits, yb)

                loss = losses.mean()
                k = n = int(losses.numel())
                if epoch >= ohwarmup:
                    k = min(n, int(ohkeep)) if ohkeep >= 1.0 else max(1, int(ohkeep * n))
                    hard_losses, _ = torch.topk(losses, k)
                    ohmean = hard_losses.mean()
                    loss = ohalpha * ohmean + (1.0 - ohalpha) * loss

                selected_total += k

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(helper.model.parameters(), clip)
                opt.step()

                batch_size_actual = int(yb.size(0))
                total       += batch_size_actual
                total_loss  += loss.detach() * k

                probs = logits.softmax(dim=1)
                preds = probs.argmax(dim=1)
                correct_total += (preds == yb).sum()

            helper.retrained = True

            train_acc = correct_total.item() / total if total > 0 else 0.0
            avg_loss  = total_loss.item() / (selected_total if selected_total > 0 else 1)
            scheduler.step(train_acc)
            current_lr = opt.param_groups[0]['lr']

            print(dt.now() - now)
            print(f'Epoch {epoch+1}/{epochs}',
                  f'loss={avg_loss:.4f}',
                  f'acc={train_acc:.3f}',
                  f'lr={current_lr:.2e}',
                  f'wd={wd:.1e}',
                  )

            # ----------------------------------------------------------NOVO
            # Loga métricas por época no MLflow
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_acc":  train_acc,
                "lr":         current_lr,
            }, step=epoch)
            # --------------------------------------------------------- /NOVO

            if callable(check): check()


def merge(model_a: nn.Module, model_b: nn.Module, alpha: float = 0.6) -> nn.Module:
    """Merge two `nn.Module` instances (same architecture) by averaging weights."""
    if not isinstance(model_a, nn.Module) or not isinstance(model_b, nn.Module):
        raise ValueError("merge expects two nn.Module instances")

    out_model = copy.deepcopy(model_a)
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    merged = {}
    for k in sd_a.keys():
        if k not in sd_b:
            raise RuntimeError(f"Key {k} missing in model_b")
        a = sd_a[k].cpu()
        b = sd_b[k].cpu()
        if a.shape != b.shape:
            raise RuntimeError(f"Shape mismatch for {k}: {a.shape} vs {b.shape}")
        merged[k] = (alpha * a) + ((1.0 - alpha) * b)

    out_model.load_state_dict(merged)
    return out_model
