"""Busy work, inherited leases and one worker per node survive budget reductions."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from replay_worker_budget import idle_retirement_candidates


def worker(address, busy=False, long=False):
    return SimpleNamespace(address=address, processing={'task'} if busy else set(), long_running={'task'} if long else set())


def test_busy_worker_survives_and_idle_sibling_can_retire():
    a,b=worker('a',True),worker('b')
    assert idle_retirement_candidates([a,b],{'a':'node','b':'node'}) == [b]


def test_each_idle_node_retains_one_worker():
    rows=[worker(c) for c in 'abcd']
    result=idle_retirement_candidates(rows,dict(a='one',b='one',c='two',d='two'))
    assert [w.address for w in result] == ['b','d']


def test_unknown_node_is_retained():
    a,b=worker('a'),worker('b')
    assert idle_retirement_candidates([a,b],{'a':'node'}) == []


def test_orphan_with_inherited_source_lease_is_retained():
    a,b=worker('a'),worker('b')
    assert idle_retirement_candidates([a,b],{'a':'node','b':'node'},['b']) == [a]


def test_long_running_worker_is_never_returned():
    a,b=worker('a',True),worker('b',long=True)
    assert idle_retirement_candidates([a,b],{'a':'node','b':'node'}) == []


def test_new_work_started_after_mapping_is_protected():
    a,b=worker('a'),worker('b')
    b.processing.add('new-task')
    assert idle_retirement_candidates([a,b],{'a':'node','b':'node'}) == [a]
