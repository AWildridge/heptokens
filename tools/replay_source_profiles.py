"""Choose validated detector layouts while bounding each worker's CPU claims."""
import math


def select_detectors(source_capacity, workers, paired_speedup, min_advantage=.05, io_saturated=False):
    if type(source_capacity) is not int or type(workers) is not int or source_capacity < 0 or workers < 0:
        raise ValueError('Source capacity and worker counts must be nonnegative integers')
    if not math.isfinite(paired_speedup) or paired_speedup <= 0 or not 0 <= min_advantage < 1:
        raise ValueError('Invalid measured layout performance')
    if io_saturated or not source_capacity or not workers:
        return 1
    single_capacity = min(source_capacity, 4 * workers)
    paired_capacity = min(source_capacity, 2 * workers)
    return 2 if paired_capacity * paired_speedup > single_capacity * (1 + min_advantage) else 1


def next_detectors(active_costs, preferred, cores=4, max_sources=4):
    costs = list(active_costs)
    if preferred not in [1, 2] or any(type(c) is not int or c not in [1, 2] for c in costs):
        raise ValueError('Unsupported detector CPU claim')
    if sum(costs) > cores:
        raise ValueError('Existing claims exceed worker CPU budget')
    if len(costs) >= max_sources or sum(costs) + preferred > cores:
        return None
    return preferred


def plan_source(headrooms, node, size):
    """Avoid predictably rejected submissions; this is not a disk reservation."""
    if type(size) is not int or size <= 0:
        raise ValueError('A positive source budget is required')
    if node not in headrooms or headrooms[node] < size:
        return False
    headrooms[node] -= size
    return True
