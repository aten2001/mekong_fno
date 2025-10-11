# tests/test_perf.py
import os, time, numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.model_fno import SeasonalFNO1D  # imports TF under CPU setting

# Default threshold: 2.0s
MAX_S = float(os.environ.get("MAX_LATENCY_SEC", "2.0"))

def test_single_forward_latency():
    """
    a single forward pass (no weight loading;
    time the second call after graph building)
    on CPU should be < MAX_S seconds
    """
    # Same as training: L=120, input_features=6
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6)
    x = np.zeros((1, 120, 6), dtype=np.float32)

    # warm-up
    _ = model(x, training=False)

    # timed forward
    t0 = time.perf_counter()
    _ = model(x, training=False)
    dt = time.perf_counter() - t0

    assert dt < MAX_S, f"single forward too slow: {dt:.3f}s (limit={MAX_S}s)"
