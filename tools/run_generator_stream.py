"""Bound concurrent remote CMS readers with a process-owned canonical I/O slot."""
import json
import subprocess
import sys
from pathlib import Path
from replay_io_control import controlled_slot

with controlled_slot(sys.argv[2], sys.argv[1], 'max_streams', 16):
    subprocess.run(json.loads(Path(sys.argv[3]).read_text()), check=True, timeout=3600)
