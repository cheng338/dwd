"""Fresh-process regressions for optional conic-import logging ownership."""

import importlib.util
from pathlib import Path

import dwd
import pytest


def _support():
    path = Path(__file__).with_name("optional_import_probe.py")
    spec = importlib.util.spec_from_file_location("_dwd_optional_import_test_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = _support()
CASE_IDS = [row["id"] for row in SUPPORT.cases()]


def selected_case_ids(cvxpy_available):
    """Only cold injected absence is executable without the optional package."""
    return [row["id"] for row in SUPPORT.cases()
            if cvxpy_available or (row["phase"] == "cold" and row["error"] == "missing")]


# Discovery does not import CVXPY. A discoverable but broken installation is
# deliberately exercised and fails visibly; it is not treated as absent.
SELECTED_CASE_IDS = selected_case_ids(importlib.util.find_spec("cvxpy") is not None)
CASES = [pytest.param(case_id, marks=() if case_id in SELECTED_CASE_IDS else
                     pytest.mark.skip(reason="Requires the optional CVXPY package for preload or successful import"))
         for case_id in CASE_IDS]


@pytest.fixture(scope="module")
def observations(tmp_path_factory):
    root = Path(dwd.__file__).resolve().parent.parent
    output = tmp_path_factory.mktemp("optional-import") / "observations.json"
    panel = SUPPORT.run_panel(root, output, workers=4, case_ids=SELECTED_CASE_IDS)
    assert panel["all_children_completed"], panel
    return {row["case"]["id"]: row for row in panel["cases"]}


@pytest.mark.parametrize("case_id", CASES)
def test_optional_import_preserves_application_logging_and_original_exception(case_id, observations):
    row = observations[case_id]
    assert row["actual_exit"] == 0 and not row["timed_out"]
    observed = row["observation"]
    case = row["case"]
    assert observed["status"] == "observed"
    assert observed["dwd_origins_selected"]
    assert observed["root_state_unchanged"], observed
    later = observed["later"]
    marker = "probe: subsequent application INFO"
    if case["handler"] == "configured":
        assert later["configured_handler_received_info"]
        assert not later["requested_destination_received_info"]
        assert later["root_state"] == observed["root_before"]
        assert later["custom_text"] == "CUSTOM:INFO:" + marker + "\n"
        assert later["stream_text"] == later["file_text"] == ""
    else:
        assert later["requested_destination_received_info"]
        assert later["custom_text"] == ""
        expected = "APP:INFO:" + marker + "\n"
        if case["later"] == "file":
            assert later["file_text"] == expected and later["stream_text"] == ""
        else:
            assert later["stream_text"] == expected and later["file_text"] == ""
    if case["error"] == "none":
        assert observed["target_imported"] and observed["caught"] is None
        assert all(observed["target_exports"].values())
        assert observed["import_calls"] == []
    else:
        assert not observed["target_imported"]
        assert observed["caught_is_injected"]
        before, after = observed["injected_before"], observed["caught"]
        for field in ("id", "type", "name", "message", "cause_id", "cause_type", "cause_message", "suppress_context"):
            assert after[field] == before[field], (field, observed)
        assert after["notes"] == before["notes"]
        assert len(observed["import_calls"]) == 1
        assert observed["log_records"] == []
        assert observed["import_stderr"] == observed["custom_import_text"] == ""
