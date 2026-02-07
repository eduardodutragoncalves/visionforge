import logging
import subprocess
from typing import List, Dict, Any

logger = logging.getLogger("VisionForge.GPUStats")

QUERY_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu"
]


def get_gpu_stats() -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(QUERY_FIELDS)}",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError as exc:
        logger.warning("nvidia-smi failed: %s", exc)
        return []

    stats = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue
        try:
            stats.append({
                "index": int(parts[0]),
                "name": parts[1],
                "utilization": int(parts[2]),
                "memory_used": int(parts[3]),
                "memory_total": int(parts[4]),
                "temperature": int(parts[5])
            })
        except ValueError:
            continue
    return stats
