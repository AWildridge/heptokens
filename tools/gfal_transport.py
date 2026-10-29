"""Selectable GFAL transport with one reusable, isolated context per thread."""
import errno
import os
import subprocess
import threading
import time
import json
from pathlib import Path
from contextlib import nullcontext

_contexts = threading.local()


def run(*command):
    if os.environ.get('HEPTOKENS_GFAL_BACKEND', 'cli') == 'cli':
        return subprocess.run(command, check=True, capture_output=True, text=True,
                              timeout=1800).stdout.strip()
    if os.environ['HEPTOKENS_GFAL_BACKEND'] != 'context':
        raise ValueError('Unknown GFAL backend')
    import gfal2
    if not hasattr(_contexts, 'value'):
        _contexts.value = gfal2.creat_context()
    context = _contexts.value
    try:
        if command[0] == 'gfal-sum' and len(command) == 3 and command[2] == 'ADLER32':
            return command[1] + ' ' + context.checksum(command[1], 'ADLER32')
        if command[0] != 'gfal-copy':
            raise ValueError('Unsupported command for GFAL context backend')
        options = list(command[1:-2])
        create_parent = '-p' in options
        if create_parent:
            options.remove('-p')
        if options != ['--checksum', 'ADLER32']:
            raise ValueError('Unsupported or unsafe GFAL copy options')
        parameters = context.transfer_parameters()
        parameters.create_parent = create_parent
        parameters.overwrite = False
        parameters.timeout = 1800
        parameters.set_checksum(gfal2.checksum_mode.both, 'ADLER32', '')
        kind='upload' if command[-2].startswith('file:') else 'readback'
        limits=os.environ.get('HEPTOKENS_IO_LIMITS')
        slots=os.environ.get('HEPTOKENS_IO_SLOTS')
        metrics=os.environ.get('HEPTOKENS_IO_METRICS')
        began=time.time()
        def progress(status):
            if metrics:
                from replay_remote_bundle import atomic
                atomic(Path(metrics).with_suffix('.current.json'),{'state':status,'kind':kind,'time':time.time()})
        progress('waiting')
        if limits and slots:
            from replay_io_control import controlled_slot
            scope=controlled_slot(Path(slots)/kind,limits,'max_uploads' if kind=='upload' else 'max_readbacks',8)
        else:scope=nullcontext()
        with scope:
            started=time.time();progress('copying')
            context.filecopy(parameters, command[-2], command[-1])
            ended=time.time()
        if metrics:
            from urllib.parse import urlsplit,unquote
            local=command[-2] if kind=='upload' else command[-1]
            size=Path(unquote(urlsplit(local).path)).stat().st_size
            with Path(metrics).open('a') as stream:
                stream.write(json.dumps({'kind':kind,'started':started,'finished':ended,'wait_seconds':started-began,'copy_seconds':ended-started,'bytes':size,'status':'copied-checksum-pending'})+'\n')
        progress('idle')
        return ''
    except gfal2.GError as error:
        if 'progress' in locals():progress('failed')
        if 'metrics' in locals() and metrics:
            with Path(metrics).open('a') as stream:
                stream.write(json.dumps({'kind':kind,'finished':time.time(),'status':'failed','error':str(error)})+'\n')
        message = str(error)
        if error.code == errno.ENOENT:
            message = 'No such file: ' + message
        raise subprocess.CalledProcessError(1, command, stderr=message) from error
