from scripts.evaluate_l7_phase123_diagnostics import (
    _phase_path_class,
    _summarize_rows,
)


def test_phase_path_class_distinguishes_phase4_failure():
    assert _phase_path_class({1: {"ok": False}}) == "phase1_failed"
    assert _phase_path_class({1: {"ok": True}, 2: {"ok": False}}) == "phase1_ok_phase2_failed"
    assert _phase_path_class({1: {"ok": True}, 2: {"ok": True}, 3: {"ok": False}}) == "phase12_ok_phase3_failed"
    assert _phase_path_class({1: {"ok": True}, 2: {"ok": True}, 3: {"ok": True}}) == "phase123_ok_phase4_not_run"
    assert (
        _phase_path_class({1: {"ok": True}, 2: {"ok": True}, 3: {"ok": True}, 4: {"ok": False}})
        == "phase123_ok_phase4_failed"
    )
    assert (
        _phase_path_class({
            1: {"ok": True},
            2: {"ok": True},
            3: {"ok": True},
            4: {"ok": False, "failed_stage_index": 3},
        })
        == "phase123_ok_full4_phase3_failed"
    )
    assert (
        _phase_path_class({1: {"ok": True}, 2: {"ok": True}, 3: {"ok": True}, 4: {"ok": True}})
        == "phase1234_solved"
    )


def test_summarize_rows_reports_phase4_contract():
    rows = [
        {
            "scenario": "solved.json",
            "solved123": True,
            "solved1234": True,
            "failedAt": None,
            "phasePathClass": "phase1234_solved",
            "phase1Prefix": {"ok": True},
            "phase2Prefix": {"ok": True},
            "phase3Prefix": {"ok": True},
            "phase4Prefix": {"ok": True},
            "phase2Actual": {"canEnterPhase3": True},
            "phase4Actual": {"hookCount": 2},
        },
        {
            "scenario": "phase4_failed.json",
            "solved123": True,
            "solved1234": False,
            "failedAt": "final_exact_settle_and_cleanup",
            "phasePathClass": "phase123_ok_phase4_failed",
            "phase1Prefix": {"ok": True},
            "phase2Prefix": {"ok": True},
            "phase3Prefix": {"ok": True},
            "phase4Prefix": {"ok": False},
            "phase2Actual": {"canEnterPhase3": True},
            "phase3Actual": {"hookCount": 3, "depotGoalCompletionRate": 1.0},
        },
    ]

    summary = _summarize_rows(rows)

    assert summary["phase123_ok_count"] == 2
    assert summary["phase1234_ok_count"] == 1
    assert summary["phase4_ok_count"] == 1
    assert summary["phase_path_class_distribution"] == {
        "phase1234_solved": 1,
        "phase123_ok_phase4_failed": 1,
    }
    assert summary["failed_scenarios"] == [
        {
            "scenario": "phase4_failed.json",
            "failedAt": "final_exact_settle_and_cleanup",
            "phase1": {"ok": True},
            "phase2": {"ok": True},
            "phase3": {"ok": True},
            "phase4": {"ok": False},
            "phasePathClass": "phase123_ok_phase4_failed",
        }
    ]
