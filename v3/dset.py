from __future__ import annotations
from pathlib import Path
import random
from glob import glob
from typing import List, Callable
from torch.utils.data import Dataset

def _pics_from_dir(d: str | Path | None, /) -> List[str]:
	if not d: return []
	if isinstance(d, str): d = Path(d)
	if not d.is_dir(): return []
	files = glob(str(d / '*'))
	return [p for p in files if p.lower().endswith(('.jpg', '.png', '.jpeg'))]

class DirDataset(Dataset):
	def __init__(self, 
		dir: str | Path | DirDataset, 
		sub: str | None = None,
		/, *,
		transform: Callable | None = None, 
		shuffle: bool = True,
		limit: int | None = None,
	):
		self.transform = transform
		self.samples: List[tuple[str, int]] = []
		if isinstance(dir, DirDataset):
			self.samples = list(dir.samples)
			if limit is not None:
				limit = int(limit)
				self.samples = self.samples[:limit]
			if shuffle: random.shuffle(self.samples)
			return
		elif dir is None: dir = Path.cwd()
		elif isinstance(dir, str | Path): dir = Path(dir)
		if sub is not None:
			s = dir / sub
			if s.is_dir(): dir = s
		reals = _pics_from_dir(dir / 'real')
		fakes = _pics_from_dir(dir / 'fake')
		if limit is not None:
			limit = int(limit)
			reals = reals[:limit]
			fakes = fakes[:limit]
		self.samples += [(p, 0) for p in fakes]
		self.samples += [(p, 1) for p in reals]
		

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		path, label = self.samples[idx]
		img = self.transform(path) if self.transform else path
		return img, label

	def __add__(self, # type: ignore[override]
		other: DirDataset,
	) -> DirDataset: 
		if not isinstance(other, DirDataset):
			return NotImplemented
		# Create instance without calling __init__ to avoid re-scanning dirs
		new = object.__new__(DirDataset)
		new.transform = self.transform
		# Merge samples (left then right) and shuffle
		new.samples = list(self.samples) + list(other.samples)
		random.shuffle(new.samples)
		return new


