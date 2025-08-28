# --- 保留你原本的 StepTimers 與 timed 定義 ---
import time
from contextlib import contextmanager
from collections import defaultdict
import torch

class StepTimers:
    """Measure CPU wall time and GPU kernel time per named section."""
    def __init__(self, device: torch.device):
        self.device = device
        self.has_cuda = (device.type == "cuda")
        self.cpu_t0 = {}
        self.cpu_ms = defaultdict(float)
        self.gpu_start = {}
        self.gpu_end = {}
        self.gpu_ms = defaultdict(float)

    def _new_event(self):
        return torch.cuda.Event(enable_timing=True)

    def start(self, name: str):
        self.cpu_t0[name] = time.perf_counter()
        if self.has_cuda:
            s = self._new_event(); e = self._new_event()
            self.gpu_start[name] = s; self.gpu_end[name] = e
            s.record()  # record on default stream

    def stop(self, name: str):
        # CPU
        t1 = time.perf_counter()
        self.cpu_ms[name] += (t1 - self.cpu_t0[name]) * 1000.0
        # GPU
        if self.has_cuda and name in self.gpu_end:
            self.gpu_end[name].record()

    def finalize(self):
        """Call once at the end of the step to collect GPU timings."""
        if self.has_cuda:
            torch.cuda.synchronize()  # wait all recorded events
            for name, end_ev in self.gpu_end.items():
                start_ev = self.gpu_start[name]
                self.gpu_ms[name] += start_ev.elapsed_time(end_ev)  # in ms

    def to_dict(self):
        # Prefer GPU time if available; include both for參考
        out = {}
        for k in set(list(self.cpu_ms.keys()) + list(self.gpu_ms.keys())):
            out[k] = {
                "gpu_ms": float(self.gpu_ms.get(k, 0.0)),
                "cpu_ms": float(self.cpu_ms.get(k, 0.0)),
            }
        return out

@contextmanager
def timed(timers: StepTimers, name: str):
    timers.start(name)
    try:
        yield
    finally:
        timers.stop(name)
