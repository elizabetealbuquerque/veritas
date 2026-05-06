import numpy as np
from typing import List, Optional, Sequence
from pathlib import Path
from PIL import Image as pil

import torch
from torch import Tensor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try: from . import helper
except ImportError:
	import helper

__all__ = ['heatmap', 'predict', 'evaluate_folder']

@helper.timer
def heatmap(img_rgb) -> tuple[float, pil.Image]:
	# Predict probability of "real" and generate a Grad-CAM heatmap for the input pil
	# This function uses the difference between the "real" and "fake" class activations to produce a single heatmap.
	if isinstance(img_rgb, str):
		img_rgb = pil.open(img_rgb).convert('RGB')
	tr = helper.transform()
	device = helper.best_device()
	tensor = tr(img_rgb)
	tensor = tensor.unsqueeze(0).to(device, non_blocking=True)  # type: ignore

	helper.model.eval()
	helper.model.to(device)
	with torch.no_grad():
		logits = helper.expand(helper.model(tensor))
		probs = logits.softmax(dim=1)

	# find last Conv2d as target layer
	layer = getattr(helper.model, 'conv_head', None)
	if layer is None:
		for m in reversed(list(helper.model.modules())):
			if isinstance(m, torch.nn.Conv2d):
				layer = m
				break
		if layer is None:
			raise RuntimeError('target layer for Grad-CAM not found')

	def cam_target(label: str , cam: GradCAM):
		tgt = ClassifierOutputTarget(helper.LABEL[label])
		greyscale = cam(input_tensor=tensor, targets=[tgt])[0] # type: ignore
		return np.array(pil.fromarray(greyscale).resize(img_rgb.size))
	
	with GradCAM(model=helper.model, target_layers=[layer]) as cam:
		fake_cam_img = cam_target('fake', cam)
		real_cam_img = cam_target('real', cam)
		greyscale = fake_cam_img - real_cam_img
		min, max = greyscale.min(), greyscale.max()
		greyscale = (greyscale - min) / (max - min + 1e-8)
		
		cam_img = show_cam_on_image(np.array(img_rgb) / 255.0, greyscale, use_rgb=True)
	return probs[0, helper.LABEL['real']].item(), pil.fromarray(cam_img)

def predict(imgs_rgb: Sequence | Tensor) -> List[float]:
	# Predict probabilities of "real" for a batch of input pil images or tensors.
	# Returns a list of floats in [0,1].
	if not imgs_rgb or not len(imgs_rgb):
		return []

	device = helper.best_device()
	helper.model.eval()
	helper.model.to(device)
	# single tensor batch
	if isinstance(imgs_rgb, Tensor) and imgs_rgb.ndim == 3:
		imgs_rgb.unsqueeze_(0).unsqueeze_(0)
	
	# sequence of tensors -> assume already transformed
	tr = helper.transform()
	tensors = []
	for t in imgs_rgb:
		if not isinstance(t, Tensor):
			t = tr(t)
		if t.ndim == 4 and t.shape[0] == 1:
			t = t.squeeze(0)
		tensors.append(t)
	batch = torch.stack(tensors).to(device, 
		non_blocking=True)
	with torch.no_grad():
		logits = helper.expand(helper.model(batch))
		probs = logits.softmax(dim=1)[:, helper.LABEL['real']]
		probs = probs.cpu().tolist()
	return probs

@helper.timer
def evaluate_folder(
	test_dir: str | Path | helper.DirDataset, 
	batch_size: int = 64, 
	thresh: float = 0.7,
	limit: Optional[int] = None,
) -> float:
	# Run prediction on all images in the specified folder (with 'real' and 'fake' subfolders) 
	# This is a simple evaluation function 
	# that processes the test set in batches and prints out the percentages of wrong/sure/dunno predictions based on the specified threshold.
	test_dir = helper.DirDataset(test_dir, 'test',
		shuffle=True, limit=limit, transform=helper.transform(train=False))
	probs: list = [] 
	ylabels: list = []
	for i in range(0, len(test_dir), batch_size):
		batch = [test_dir[j] for j in range(i, min(i + batch_size, len(test_dir)))]
		imgs, labels = zip(*batch)
		if not imgs: continue
		probs.extend(predict(imgs))
		ylabels.extend(labels[:len(imgs)])
	return helper.compare(probs, ylabels, thresh=thresh)