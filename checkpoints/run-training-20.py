import ctypes
from pathlib import Path
import subprocess
import sys

checkpoint_dir = Path(__file__).resolve().parent
repo = checkpoint_dir.parent
keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
keep_awake.argtypes = [ctypes.c_uint]
keep_awake.restype = ctypes.c_uint
continuous = 0x80000000
if not keep_awake(continuous | 1):
    raise SystemExit("Could not activate Windows idle sleep hold")

try:
    print("Windows idle sleep hold active", flush=True)
    with subprocess.Popen(
        [sys.executable, "-u", str(repo / "cnn" / "train.py")], cwd=repo
    ) as training:
        print(f"Training PID {training.pid}. Starting 20 epochs from fresh weights.", flush=True)
        code = training.wait()

    if code:
        raise SystemExit(f"Training process exited with code {code}")
    checkpoints = [checkpoint_dir / f"cnn_epoch_{epoch}.pth" for epoch in range(1, 21)]
    missing = [path.name for path in checkpoints if not path.is_file() or not path.stat().st_size]
    if missing:
        raise SystemExit(f"Missing or empty checkpoints: {missing}")
    import torch
    torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    print("Training completed successfully. Verified 20 checkpoints and loaded the final checkpoint.", flush=True)
finally:
    keep_awake(continuous)
    print("Windows idle sleep hold released", flush=True)
