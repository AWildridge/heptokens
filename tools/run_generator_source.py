"""Isolate a complete source transaction from the Dask worker process.

The child owns the source lease. If the worker is restarted, it can finish and
commit; a replacement worker receives source-busy and waits instead of failing.
"""
import json
from pathlib import Path
import sys
import traceback
from run_generator_backfill import atomic, run_task

request=json.loads(Path(sys.argv[1]).read_text())
try:
    result=run_task(request['config'],request['source'],request.get('limit',-1),request.get('cached'),True)
except Exception as error:
    result=dict(status='source-failed',task=int(request['source']['task']),
                retryable=not isinstance(error,ValueError),error_type=type(error).__name__,error=traceback.format_exc())
atomic(request['result'],result)
