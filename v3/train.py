import torch
from datetime import datetime as dt
from pathlib import Path
from typing import Callable
try: from . import helper
except ImportError:
	import helper

__all__ = ['train']

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
):
	# Train for two-class detection: 0=fake, 1=real.
	# Supports optional Online Hard Example Mining (OHEM).
	# training transform (with augmentations)
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
		threshold=0.005, min_lr=1e-5, cooldown=0, )
	helper.model.to(device)
	centropy = torch.nn.CrossEntropyLoss(reduction='none')
	print('Starting training...')
	for epoch in range(epochs):
		if callable(check): check()
		helper.model.train()
		now = dt.now()
		total_loss = torch.tensor(0.0, device=device)
		total = 0
		correct_total = torch.tensor(0, device=device)
		selected_total = 0
		for batch in train_loader:
			xb, yb = batch
			xb = xb.to(device, non_blocking=True)
			yb = yb.long().to(device, non_blocking=True)

			logits = helper.expand(helper.model(xb))
			losses = centropy(logits, yb)

			# Apply OHEM: pick top-k hardest samples per batch (after warmup)
			if epoch >= ohwarmup:
				n = int(losses.numel())
				if ohkeep >= 1:
					k = min(n, int(ohkeep))
				else:
					k = max(1, int(ohkeep * n))
				hard_losses, _ = torch.topk(losses, k)
				# Weighted mix between OHEM mean and full-batch mean
				batch_mean = losses.mean()
				ohmean = hard_losses.mean()
				loss = ohalpha * ohmean + (1.0 - ohalpha) * batch_mean
				n_selected = k
				selected_total += k
			else:
				loss = losses.mean()
				n_selected = int(losses.numel())
				selected_total += n_selected

			opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(helper.model.parameters(), clip)
			opt.step()

			batch_size_actual = int(yb.size(0))
			total += batch_size_actual
			total_loss += loss.detach() * n_selected
			probs = logits.softmax(dim=1)
			preds = probs.argmax(dim=1)
			correct_total += (preds == yb).sum()
		helper.retrained = True
		train_acc = correct_total.item() / total if total > 0 else 0.0
		avg_loss = total_loss.item() / (selected_total if selected_total > 0 else 1)
		scheduler.step(train_acc)
		current_lr = opt.param_groups[0]['lr']
		
		print(dt.now() - now)
		print(f'Epoch {epoch+1}/{epochs}', 
			f'loss={avg_loss:.4f}',
			f'acc={train_acc:.3f}',
			f'lr={current_lr:.2e}',
			f'wd={wd:.1e}',
		)
	if callable(check): check()
	