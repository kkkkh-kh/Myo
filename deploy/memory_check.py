import gc
from statistics import mean, pstdev
from typing import Dict, Sequence

import psutil


class MemoryProfiler:
    """Profile incremental RSS memory during inference."""

    def __init__(self) -> None:
        self.process = psutil.Process()

    def _rss_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def measure_inference_memory(self, pipeline, test_inputs: Sequence[str]) -> Dict[str, float]:
        if not test_inputs:
            raise ValueError("test_inputs must not be empty")

        gc.collect()
        pipeline._ensure_loaded()
        gc.collect()
        baseline_mb = self._rss_mb()
        memory_samples = []

        for index in range(100):
            text = test_inputs[index % len(test_inputs)]
            pipeline.translate(text)
            current_mb = max(0.0, self._rss_mb() - baseline_mb)
            memory_samples.append(current_mb)

        peak_mb = max(memory_samples)
        mean_mb = mean(memory_samples)
        std_mb = pstdev(memory_samples) if len(memory_samples) > 1 else 0.0
        limit_mb = float(getattr(pipeline, "memory_limit_mb", 60.0))
        passes_over_limit = sum(sample > limit_mb for sample in memory_samples)
        return {
            "peak_mb": round(peak_mb, 4),
            "mean_mb": round(mean_mb, 4),
            "std_mb": round(std_mb, 4),
            "passes_over_limit": int(passes_over_limit),
        }

    @staticmethod
    def assert_under_limit(peak_mb: float, limit_mb: float = 60) -> None:
        if peak_mb > limit_mb:
            raise RuntimeError(f"推理内存超限：{peak_mb:.2f}MB > {limit_mb:.2f}MB")
