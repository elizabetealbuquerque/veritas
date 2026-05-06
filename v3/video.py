from __future__ import annotations

from typing import Sequence, List, Dict, Optional
import cv2, math
import numpy as np

try: from . import helper, predict
except ImportError:
    import helper, predict

def predict_video(
    video_path: str,
    num_frames: int = 32,
    batch_size: int = 32,
    heatmap_frames: Optional[Sequence[int]] = None,
    thresh: float = 0.6,
    random_seed: Optional[int] = None,
) -> Dict:
    # Open video
    # Note: OpenCV's VideoCapture can be unreliable with certain formats/codecs, and
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rng = np.random.default_rng(random_seed)

    sampled: List[tuple] = []  # list of (orig_idx, timestamp, PIL.Image)

    if total_frames > 0:
        # Choose `num_frames` indices with stratified randomness across video
        n = min(max(1, int(num_frames)), total_frames)
        seg = float(total_frames) / n
        chosen = []
        for i in range(n):
            start = int(math.floor(i * seg))
            end = int(math.floor((i + 1) * seg)) - 1
            if end < start:
                end = start
            # sample uniformly within the segment
            idx_choice = int(rng.integers(start, end + 1))
            chosen.append(idx_choice)
        # Ensure unique and sort (temporal order)
        chosen = sorted(sorted(set(chosen))[:n])

        # Read only the chosen frames by seeking
        for fi in chosen:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            pil_img = helper.frame_bgr_to_pil(frame)
            timestamp = (fi / fps) if fps > 0 else None
            sampled.append((fi, timestamp, pil_img))
    else:
        # Unknown total frames: read all frames into memory then sample
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            idx += 1
        cap.release()
        L = len(frames)
        if L == 0:
            return {'frames': [], 'stats': {'mean': 0.0, 'median': 0.0, 'pct_real': 0.0, 'n_frames': 0, 'total_frames_in_file': 0}, 'heatmaps': {}}
        n = min(max(1, int(num_frames)), L)
        seg = float(L) / n
        chosen = []
        for i in range(n):
            start = int(math.floor(i * seg))
            end = int(math.floor((i + 1) * seg)) - 1
            if end < start:
                end = start
            idx_choice = int(rng.integers(start, end + 1))
            chosen.append(idx_choice)
        chosen = sorted(sorted(set(chosen))[:n])
        for fi in chosen:
            frame = frames[fi]
            pil_img = helper.frame_bgr_to_pil(frame)
            timestamp = (fi / fps) if fps > 0 else None
            sampled.append((fi, timestamp, pil_img))

    # If cap wasn't released in the total_frames branch, release now
    try:
        cap.release()
    except Exception:
        pass

    results: List[Dict] = []
    heatmaps_out: Dict[int, object] = {}

    # Batch predict
    batch_imgs: List = []
    batch_meta: List = []
    for i, (orig_idx, ts, pil_img) in enumerate(sampled):
        batch_imgs.append(pil_img)
        batch_meta.append((orig_idx, ts))
        if len(batch_imgs) >= batch_size:
            probs = predict.predict_batch(batch_imgs)
            for (oidx, t), p in zip(batch_meta, probs):
                results.append({'frame_index': oidx, 'time': t, 'prob': float(p)})
            batch_imgs = []
            batch_meta = []

    # Remaining
    if batch_imgs:
        probs = predict.predict_batch(batch_imgs)
        for (oidx, t), p in zip(batch_meta, probs):
            results.append({'frame_index': oidx, 'time': t, 'prob': float(p)})

    # Stats
    probs_only = [r['prob'] for r in results]
    n = len(probs_only)
    mean = float(np.mean(probs_only)) if n else 0.0
    median = float(np.median(probs_only)) if n else 0.0
    pct_real = float(sum(1 for p in probs_only if p >= thresh) * 100.0 / n) if n else 0.0

    # Heatmaps for selected sampled indices (relative to sampled order)
    if heatmap_frames:
        # map sampled-order index -> PIL image
        for rel_idx in heatmap_frames:
            if rel_idx < 0 or rel_idx >= len(sampled):
                continue
            orig_idx, ts, pil_img = sampled[rel_idx]
            try:
                p, heat = predict.predict_and_heatmap(pil_img)
                heatmaps_out[orig_idx] = {'prob': float(p), 'heatmap': heat}
            except Exception as e:
                heatmaps_out[orig_idx] = {'error': str(e)}

    return {
        'frames': results,
        'stats': {
            'mean': mean,
            'median': median,
            'pct_real': pct_real,
            'n_frames': n,
            'total_frames_in_file': total_frames,
        },
        'heatmaps': heatmaps_out,
    }
