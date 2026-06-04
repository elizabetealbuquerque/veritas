import numpy as np
from typing import List, Optional, Sequence
from pathlib import Path
from PIL import Image as pil
from datetime import datetime as dt
import torch
from torch import Tensor
import mlflow                          # << NOVO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try: from . import helper
except ImportError:
    import helper

__all__ = ['heatmap', 'predict', 'evaluate_folder']

@helper.timer
def heatmap(img_rgb) -> tuple[float, pil.Image]:
    # Predict probability of "real" and generate a Grad-CAM heatmap.
    if isinstance(img_rgb, str):
        img_rgb = pil.open(img_rgb).convert('RGB')

    tr     = helper.transform()
    device = helper.best_device()
    tensor = tr(img_rgb)
    tensor = tensor.unsqueeze(0).to(device, non_blocking=True)  # type: ignore

    helper.model.eval()
    helper.model.to(device)

    with torch.no_grad():
        logits = helper.expand(helper.model(tensor))
        probs  = logits.softmax(dim=1)

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
        greyscale = cam(input_tensor=tensor, targets=[tgt])[0]  # type: ignore
        return np.array(pil.fromarray(greyscale).resize(img_rgb.size))

    with GradCAM(model=helper.model, target_layers=[layer]) as cam:
        fake_cam_img = cam_target('fake', cam)
        real_cam_img = cam_target('real', cam)

    greyscale = fake_cam_img - real_cam_img
    min, max  = greyscale.min(), greyscale.max()
    greyscale = (greyscale - min) / (max - min + 1e-8)
    cam_img   = show_cam_on_image(np.array(img_rgb) / 255.0, greyscale, use_rgb=True)

    return probs[0, helper.LABEL['real']].item(), pil.fromarray(cam_img)


def predict(imgs_rgb: Sequence | Tensor) -> List[float]:
    # Predict probabilities of "real" for a batch of PIL images or tensors.
    if imgs_rgb is None:
        return []

    device = helper.best_device()
    helper.model.eval()
    helper.model.to(device)

    if not isinstance(imgs_rgb, Tensor):
        tr       = helper.transform(train=False)
        imgs_rgb = [tr(img) for img in imgs_rgb]
        imgs_rgb = torch.stack(imgs_rgb)

    while imgs_rgb.ndim <= 3:
        imgs_rgb.unsqueeze_(0)

    batch = imgs_rgb.to(device, non_blocking=True)
    with torch.no_grad():
        logits = helper.expand(helper.model(batch))
        probs  = logits.softmax(dim=1)[:, helper.LABEL['real']]

    return probs.cpu().tolist()


@helper.timer
def evaluate_folder(
    test_dir: str | Path | helper.DirDataset,
    batch_size: int = 32,
    thresh: float = 0.7,
    limit: Optional[int] = None,
) -> float:
    now = dt.now()

    test_dir = helper.DirDataset(test_dir, 'test',
                    shuffle=True, limit=limit,
                    transform=helper.transform(train=False))
    print(dt.now() - now)

    probs:   list = []
    ylabels: list = []

    for b in range(0, len(test_dir), batch_size):
        batch_imgs   = []
        batch_labels = []
        for i in range(b, min(b + batch_size, len(test_dir))):
            img, label = test_dir[i]
            batch_imgs.append(img)
            batch_labels.append(label)
        batch_imgs = torch.stack(batch_imgs)
        probs.extend(predict(batch_imgs))
        ylabels.extend(batch_labels)

    acc = helper.compare(probs, ylabels, thresh=thresh)

    # -------------------------------------------------------------- NOVO
    # Se houver um run ativo (aberto pelo train()), loga a métrica de avaliação
    if mlflow.active_run():
        mlflow.log_metric("eval_acc", acc)
    # ------------------------------------------------------------- /NOVO

    return acc
