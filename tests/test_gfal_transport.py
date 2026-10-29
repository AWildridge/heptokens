import errno
from pathlib import Path
import subprocess
import sys
import threading
import types
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import gfal_transport as transport


def test_context_reused_and_no_overwrite(monkeypatch):
    calls=[]
    class Context:
        def checksum(self,*args):calls.append(('checksum',args));return 'abc'
        def transfer_parameters(self):
            return types.SimpleNamespace(set_checksum=lambda *args:calls.append(('checked',args)))
        def filecopy(self,parameters,*args):
            assert parameters.overwrite is False
            assert parameters.create_parent is True
            calls.append(('copy',args))
    def create():calls.append(('created',));return Context()
    monkeypatch.setitem(sys.modules,'gfal2',types.SimpleNamespace(creat_context=create,checksum_mode=types.SimpleNamespace(both='both'),GError=RuntimeError))
    monkeypatch.setenv('HEPTOKENS_GFAL_BACKEND','context');monkeypatch.setattr(transport,'_contexts',threading.local())
    transport.run('gfal-sum','davs://example/a','ADLER32')
    transport.run('gfal-copy','-p','--checksum','ADLER32','file:///tmp/a','davs://example/a')
    transport.run('gfal-sum','davs://example/a','ADLER32')
    assert sum(c[0]=='created' for c in calls)==1
    with pytest.raises(ValueError):transport.run('gfal-copy','--force','file:///tmp/a','davs://example/a')


def test_missing_and_authentication_errors_remain_distinct(monkeypatch):
    class Error(Exception):
        def __init__(self,code):self.code=code
    state={'code':errno.ENOENT}
    class Context:
        def checksum(self,*args):raise Error(state['code'])
    monkeypatch.setitem(sys.modules,'gfal2',types.SimpleNamespace(creat_context=Context,GError=Error))
    monkeypatch.setenv('HEPTOKENS_GFAL_BACKEND','context');monkeypatch.setattr(transport,'_contexts',threading.local())
    with pytest.raises(subprocess.CalledProcessError) as missing:transport.run('gfal-sum','url','ADLER32')
    assert 'No such file' in missing.value.stderr
    state['code']=errno.EACCES
    with pytest.raises(subprocess.CalledProcessError) as denied:transport.run('gfal-sum','url','ADLER32')
    assert 'No such file' not in denied.value.stderr
