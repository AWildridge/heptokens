"""Separate process for canonical global download slots and bounded XRootD."""
import subprocess
import sys
from replay_io_control import controlled_slot

with controlled_slot(sys.argv[2], sys.argv[1], 'max_downloads', 16):
    subprocess.run(['/work/pixi/global/.pixi/envs/default/bin/xrdcp', '--nopbar', '--retry', '0', sys.argv[3], sys.argv[4]],
                   check=True, timeout=1800)
