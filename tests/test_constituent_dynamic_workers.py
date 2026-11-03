"""Source jobs queued before workers exist must run after resource publication."""
import sys
import time
from pathlib import Path

from distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_constituent_backfill import one_source_plugin, resource_wakeup_plugin


def test_resource_jobs_survive_dynamic_worker_startup():
    with LocalCluster(n_workers=0, threads_per_worker=1, processes=False,
                      dashboard_address=None, memory_limit='512MiB', silence_logs=40) as cluster:
        with Client(cluster) as client:
            client.register_plugin(one_source_plugin(), name='constituent-one-source')
            client.register_plugin(resource_wakeup_plugin(), name='constituent-resource-wakeup')
            future = client.submit(lambda: 'resource-job-ran',
                                   resources={'constituent_source': 1}, pure=False)
            cluster.scale(1)
            assert future.result(timeout=30) == 'resource-job-ran'
            cluster.scale(2)
            client.wait_for_workers(2, timeout=30)
            deadline = time.time() + 30
            while time.time() < deadline:
                workers = client.scheduler_info()['workers']
                if len(workers) == 2 and all(w.get('resources', {}).get('constituent_source') == 1
                                             for w in workers.values()):
                    break
                time.sleep(.1)
            else:
                raise AssertionError('New worker source resource was not published')
