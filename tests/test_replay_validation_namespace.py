import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'heptokens/tools'))


def helper():
    for path in [ROOT/'heptokens/tools/replay_worker_local.py',
                 ROOT/'purdue-setup/candidates/scratch-paired/tools/replay_worker_local.py']:
        spec=importlib.util.spec_from_file_location('namespace_subject',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module,'source_lease_name'):
            return module
    raise AssertionError('Namespace-aware helper unavailable')


def test_production_never_accepts_a_separate_writer_namespace():
    assert helper().source_lease_name('00075',None,'paired')=='00075'
    assert helper().source_lease_name('00075',None,'../../outside')=='00075'


def test_default_validation_namespace_is_preserved():
    assert helper().source_lease_name('00075',46000)=='validation-00075'


def test_independent_validation_has_separate_lease():
    assert helper().source_lease_name('00075',46000,'paired')=='validation-paired-00075'


@pytest.mark.parametrize('value',['../outside','a/b','a.b','x'*33,'☃'])
def test_validation_namespace_cannot_escape_lease_directory(value):
    with pytest.raises(ValueError):
        helper().source_lease_name('00075',46000,value)
