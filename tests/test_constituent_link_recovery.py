"""Guards against attaching native contributor labels to different detector data."""

import copy
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("ROOT")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from recover_delphes_constituent_links import (  # noqa: E402
    check_native_receipt,
    compare_compact,
)


def native_receipts():
    receipt = dict(
        input_sha256="input",
        seed=504000,
        events=1000,
        sha256={"delphes_card_CMS.tcl": "card"},
    )
    validation = dict(
        status="passed",
        input_sha256="input",
        seed=504000,
        events=1000,
        detectors={"CMS": dict(card_sha256="card", output_sha256="native")},
    )
    return validation, receipt


@pytest.mark.parametrize("field", ["input_sha256", "seed", "events", "card", "root", "status"])
def test_reject_wrong_native_simulation_provenance(field):
    validation, receipt = native_receipts()
    check_native_receipt(validation, receipt, "CMS", "native")
    changed = copy.deepcopy(validation)
    if field == "card":
        changed["detectors"]["CMS"]["card_sha256"] = "other"
    elif field == "root":
        changed["detectors"]["CMS"]["output_sha256"] = "other"
    else:
        changed[field] = "other"
    with pytest.raises(ValueError):
        check_native_receipt(changed, receipt, "CMS", "native")


def write_compact(path, p4, dtype="float32"):
    with h5py.File(path, "w") as f:
        f.create_dataset("Jet/jet_p4", data=np.asarray(p4, dtype=dtype))
        f.create_dataset("Jet/event_offsets", data=np.array([0, 1], dtype="int64"))


def test_reject_changed_compact_value_or_dtype(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    p4 = [[10.0, 0.0, 0.0, 10.0]]
    write_compact(a, p4)
    write_compact(b, p4)
    assert compare_compact(a, b) == ["Jet/event_offsets", "Jet/jet_p4"]
    write_compact(b, [[11.0, 0.0, 0.0, 11.0]])
    with pytest.raises(AssertionError):
        compare_compact(a, b)
    write_compact(b, p4, dtype="float64")
    with pytest.raises(ValueError, match="dtype"):
        compare_compact(a, b)


def test_bitwise_claim_includes_signed_zero(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    write_compact(a, [[10.0, 0.0, 0.0, 10.0]])
    write_compact(b, [[10.0, -0.0, 0.0, 10.0]])
    with pytest.raises(ValueError, match="bit pattern"):
        compare_compact(a, b)
