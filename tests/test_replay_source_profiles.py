from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from replay_source_profiles import select_detectors,next_detectors,plan_source


def test_spare_workers_and_limited_sources_use_paired_detectors():
    assert select_detectors(6,16,1.3)==2


def test_few_workers_use_four_sources_instead():
    assert select_detectors(40,5,1.3)==1


def test_changed_worker_availability_changes_layout_without_changing_data():
    assert select_detectors(20,20,1.3)==2
    assert select_detectors(20,5,1.3)==1


def test_transfer_saturation_prevents_increasing_detector_parallelism():
    assert select_detectors(6,16,1.3,io_saturated=True)==1


def test_small_or_negative_measured_advantage_keeps_single_detectors():
    assert select_detectors(6,16,1.01)==1
    assert select_detectors(6,16,.9)==1


def test_sparse_worker_placement_preserves_useful_source_slots():
    assert select_detectors(13,4,1.3)==1
    assert select_detectors(9,4,1.3)==2


@pytest.mark.parametrize('active,preferred,result',[
    ([],2,2),([2],2,2),([2,1],2,None),([2,1],1,1),([1,1],2,2),
    ([2,2],1,None),([1,1,1],1,1),([1,1,1,1],1,None)])
def test_layout_transitions_never_claim_more_than_four_cores(active,preferred,result):
    assert next_detectors(active,preferred)==result


def test_overcommitted_or_invalid_claims_fail_closed():
    with pytest.raises(ValueError):next_detectors([2,2,1],1)
    with pytest.raises(ValueError):next_detectors([0],1)
    with pytest.raises(ValueError):next_detectors([],3)


def test_empty_or_invalid_capacity():
    assert select_detectors(0,10,1.3)==1
    assert select_detectors(10,0,1.3)==1
    with pytest.raises(ValueError):select_detectors(-1,2,1.3)
    with pytest.raises(ValueError):select_detectors(1,2,float('nan'))


def test_one_dispatch_pass_does_not_spend_node_headroom_twice():
    available={'a':20,'b':10}
    assert plan_source(available,'a',8)
    assert plan_source(available,'a',8)
    assert not plan_source(available,'a',8)
    assert available=={'a':4,'b':10}


def test_unknown_or_full_nodes_are_not_given_predictably_rejected_work():
    assert not plan_source({},'missing',1)
    assert not plan_source({'full':-1},'full',1)


def test_invalid_planning_budget_fails_closed():
    with pytest.raises(ValueError):plan_source({'a':10},'a',0)
