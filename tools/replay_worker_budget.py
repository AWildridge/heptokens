"""Select idle Dask workers for retirement without losing scratch-node access."""


def idle_retirement_candidates(workers, nodes, protected=()):
    protected = set(protected)
    keepers = {}
    for worker in workers:
        node = nodes.get(worker.address)
        if node is None:
            continue
        priority = (not bool(worker.processing or worker.address in protected), worker.address)
        if node not in keepers or priority < keepers[node][0]:
            keepers[node] = priority, worker.address
    keep = {value[1] for value in keepers.values()}
    return [worker for worker in workers
            if worker.address in nodes and not worker.processing
            and not getattr(worker, 'long_running', ())
            and worker.address not in protected and worker.address not in keep]
