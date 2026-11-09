from amplitude_lab.workflows.audit_format import IssueLedger, decide, parse_audit


def test_accept_with_minor_lines_is_accepted_not_blocked():
    parsed = parse_audit(
        "VERDICT: ACCEPT\nPREVIOUS: none\nBLOCKING: none\n"
        "MINOR: M1 | tidy notation\nMINOR: M2 | add a reference\nINVALIDATE: none\n"
    )
    assert decide(parsed) == "ACCEPT"
    assert parsed.minor == ["M1 | tidy notation", "M2 | add a reference"]
    assert parsed.invalidate == "none"


def test_reject_passes_only_blocking_items_to_the_work_list():
    parsed = parse_audit(
        "VERDICT: REJECT\nPREVIOUS: none\n"
        "BLOCKING: eq3 | eq. (3) | sign flipped | re-derive the sign\n"
        "MINOR: m1 | wording\nMINOR: m2 | spacing\nINVALIDATE: none\n"
    )
    assert decide(parsed) == "REJECT"
    assert parsed.blocking == ["eq3 | eq. (3) | sign flipped | re-derive the sign"]
    assert len(parsed.minor) == 2


def test_markdown_bullets_crlf_and_continuation_lines():
    text = (
        "**VERDICT:** reject\r\n"
        "- **BLOCKING:** L1 | eq. (2) | wrong power of mu\r\n"
        "  continued on the next line | re-derive\r\n"
        "  * PREVIOUS: none\r\n"
        "invalidate: NONE.\r\n"
    )
    parsed = parse_audit(text)
    assert decide(parsed) == "REJECT"
    assert parsed.blocking == [
        "L1 | eq. (2) | wrong power of mu continued on the next line | re-derive"
    ]
    assert parsed.invalidate == "none"


def test_accept_with_blocking_item_is_contradictory():
    parsed = parse_audit("VERDICT: ACCEPT\nBLOCKING: b | eq | bad | fix\n")
    assert decide(parsed) == "CONTRADICTORY"


def test_reject_without_blocking_item_is_contradictory():
    parsed = parse_audit("VERDICT: REJECT\nBLOCKING: none\nMINOR: m | note\n")
    assert decide(parsed) == "CONTRADICTORY"


def test_garbage_text_is_contradictory_not_a_diagnostic():
    parsed = parse_audit("The derivation looks plausible to me, cheers.")
    assert parsed.verdict == "MISSING"
    assert decide(parsed) == "CONTRADICTORY"


def test_ledger_assigns_harness_ids_even_when_auditor_reuses_l1():
    ledger = IssueLedger()
    first = ledger.add_blocking(["L1 | eq (1) | wrong sign | fix"], attempt=1)
    assert first[0].harness_id == "B1"
    unmentioned = ledger.apply_previous([("B1", "OPEN", "still wrong")])
    assert unmentioned == []
    assert ledger.issues["B1"].open_rounds == 1
    second = ledger.add_blocking(["L1 | eq (9) | missing factor of i | fix"], attempt=2)
    assert second[0].harness_id == "B2"
    assert ledger.issues["B2"].text.startswith("L1")


def test_unmentioned_previous_issue_stays_open_and_counts_a_round():
    ledger = IssueLedger()
    ledger.add_blocking(["a | eq | x | fix", "b | eq | y | fix"], attempt=1)
    unmentioned = ledger.apply_previous([("B1", "RESOLVED", "fixed")])
    assert unmentioned == ["B2"]
    assert ledger.issues["B1"].status == "RESOLVED"
    assert ledger.issues["B2"].status == "OPEN"
    assert ledger.issues["B2"].open_rounds == 1


def test_accept_closes_all_open_items():
    ledger = IssueLedger()
    ledger.add_blocking(["a | eq | x | fix"], attempt=1)
    ledger.close_all()
    assert ledger.open_issues() == []
