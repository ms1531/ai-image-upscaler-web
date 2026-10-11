"""Shared Real-ESRGAN ONNX inference engine (tiled, CPU). Used by the GUI app."""
import math
import os
import numpy as np
import onnxruntime as ort
from PIL import Image


def _warmup(sess):
    """Run one real-size tile through the session. DirectML often fails here
    (at execution time, not at session creation), so this proves the GPU path
    actually works. Tile input = 256 + 2*16 pad = 288 for both models."""
    name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, 288, 288), dtype=np.float32)
    sess.run(None, {name: dummy})


def load_session(model_path, prefer_gpu=False):
    """Portable CPU-only session for maximum Windows-PC compatibility.

    This build intentionally avoids DirectML/GPU initialization because some
    office PCs fail while loading the DirectML ONNX Runtime DLLs.  It uses the
    CPUExecutionProvider only, so no GPU driver, CUDA, DirectML, or Python
    installation is required on the target PC.
    """
    opts = ort.SessionOptions()
    try:
        opts.intra_op_num_threads = max(1, min(os.cpu_count() or 4, 8))
    except Exception:
        pass
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        return sess, sess.get_inputs()[0].name, 'CPU'
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'AI CPU engine could not start: {e}')


def _run_tile(sess, in_name, tile_rgb):
    """tile_rgb: HxWx3 float32 in [0,1]. Returns 4Hx4Wx3 float32 in [0,1]."""
    t = np.transpose(tile_rgb, (2, 0, 1))[None].astype(np.float32)
    out = sess.run(None, {in_name: t})[0][0]
    out = np.transpose(out, (1, 2, 0))
    return np.clip(out, 0.0, 1.0)


class Cancelled(Exception):
    """Raised when the user stops an in-progress upscale."""


def upscale_4x(pil_img, sess, in_name, tile_size=256, overlap=16, progress_cb=None,
               cancel_event=None):
    """Tiled 4x AI upscale. Returns PIL RGB image at 4x. progress_cb(done, total)."""
    img = pil_img.convert('RGB')
    w, h = img.size
    arr = np.asarray(img).astype(np.float32) / 255.0

    pad = overlap
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    ph, pw = padded.shape[:2]

    scale = 4
    out_h, out_w = h * scale, w * scale
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)

    xs = list(range(0, pw - 2 * pad, tile_size))
    ys = list(range(0, ph - 2 * pad, tile_size))
    if xs[-1] + tile_size < pw - 2 * pad:
        xs.append(pw - 2 * pad - tile_size)
    if ys[-1] + tile_size < ph - 2 * pad:
        ys.append(ph - 2 * pad - tile_size)
    total = len(xs) * len(ys)
    done = 0

    for yy in ys:
        for xx in xs:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            tile = padded[yy:yy + tile_size + 2 * pad, xx:xx + tile_size + 2 * pad]
            up = _run_tile(sess, in_name, tile)  # ((T+2p)*4, (T+2p)*4, 3)
            # valid original-pixel extent of this tile (borders may be shorter)
            vw = min(tile_size, w - xx)
            vh = min(tile_size, h - yy)
            piece = up[pad * scale:pad * scale + vh * scale,
                       pad * scale:pad * scale + vw * scale]
            dx, dy = xx * scale, yy * scale
            canvas[dy:dy + piece.shape[0], dx:dx + piece.shape[1]] = piece
            done += 1
            if progress_cb:
                progress_cb(done, total)

    return Image.fromarray((canvas * 255.0 + 0.5).astype(np.uint8))


def upscale(pil_img, sess, in_name, scale=4, tile_size=256, overlap=16, progress_cb=None,
            cancel_event=None):
    """Full upscale to `scale` in {2,3,4}. Runs the 4x model, then resamples."""
    has_alpha = pil_img.mode == 'RGBA'
    alpha = pil_img.split()[-1] if has_alpha else None

    out4 = upscale_4x(pil_img, sess, in_name, tile_size, overlap, progress_cb,
                      cancel_event=cancel_event)

    if scale == 4:
        result = out4
    else:
        w, h = pil_img.size
        result = out4.resize((w * scale, h * scale), Image.LANCZOS)

    if has_alpha:
        w, h = pil_img.size
        a_up = alpha.resize((w * scale, h * scale), Image.LANCZOS)
        result = result.convert('RGBA')
        result.putalpha(a_up)
    return result
