"""Line-based audit reply parsing and harness-owned issue ledger.

The auditor replies with prefixed lines (VERDICT, PREVIOUS, BLOCKING, MINOR,
INVALIDATE). Only BLOCKING items can reject a step; MINOR items are kept as
polish notes. Issue ids are assigned by the harness, never trusted from the
auditor. This module performs no physics checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PREFIX = re.compile(
    r"^(VERDICT|PREVIOUS|BLOCKING|MINOR|INVALIDATE)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^(?:[-*+\u2022]\s+|\d+[.)]\s+)")
_PREVIOUS_ITEM = re.compile(
    r"^([^\s:|]+)\s*:\s*(RESOLVED|STILL OPEN|OPEN)\s*(?::\s*(.*))?$",
    re.IGNORECASE,
)


@dataclass
class ParsedAudit:
    verdict: str = "MISSING"  # ACCEPT, REJECT, or MISSING
    previous: list[tuple[str, str, str]] = field(default_factory=list)  # (id, status, evidence)
    blocking: list[str] = field(default_factory=list)
    minor: list[str] = field(default_factory=list)
    invalidate: str = "none"


def _is_none(text: str) -> bool:
    return text.strip().rstrip(".").strip().casefold() in ("", "none")


def parse_audit(text: str) -> ParsedAudit:
    """Parse a line-based audit reply. Unknown lines continue the previous item."""
    parsed = ParsedAudit()
    section = None
    for raw_line in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.replace("**", "").replace("__", "").strip()
        line = _BULLET.sub("", line).strip()
        if not line:
            section = None
            continue
        match = _PREFIX.match(line)
        if match:
            key, value = match.group(1).upper(), match.group(2).strip()
            section = key
            if key == "VERDICT":
                word = value.split()[0].upper() if value.split() else ""
                parsed.verdict = word if word in ("ACCEPT", "REJECT") else "MISSING"
            elif key == "INVALIDATE":
                parsed.invalidate = "none" if _is_none(value) else value
            elif key == "PREVIOUS" and not _is_none(value):
                _add_previous(parsed, value)
            elif key == "BLOCKING" and not _is_none(value):
                parsed.blocking.append(value)
            elif key == "MINOR" and not _is_none(value):
                parsed.minor.append(value)
            continue
        # Continuation of the previous item (blocking/minor/previous lists only).
        if section == "BLOCKING" and parsed.blocking:
            parsed.blocking[-1] += " " + line
        elif section == "MINOR" and parsed.minor:
            parsed.minor[-1] += " " + line
        elif section == "PREVIOUS" and parsed.previous:
            issue_id, status, evidence = parsed.previous[-1]
            parsed.previous[-1] = (issue_id, status, (evidence + " " + line).strip())
    return parsed


def _add_previous(parsed: ParsedAudit, value: str) -> None:
    match = _PREVIOUS_ITEM.match(value)
    if match is None:
        return
    status = "RESOLVED" if match.group(2).upper() == "RESOLVED" else "OPEN"
    evidence = (match.group(3) or "").strip()
    parsed.previous.append((match.group(1).strip(), status, evidence))


def decide(parsed: ParsedAudit) -> str:
    """Decision table: ACCEPT, REJECT, or CONTRADICTORY (re-audit the same proposal)."""
    if parsed.verdict == "ACCEPT" and not parsed.blocking:
        return "ACCEPT"
    if parsed.verdict == "REJECT" and parsed.blocking:
        return "REJECT"
    return "CONTRADICTORY"


@dataclass
class Issue:
    harness_id: str
    text: str
    first_seen_attempt: int
    status: str = "OPEN"  # OPEN or RESOLVED
    open_rounds: int = 0


class IssueLedger:
    """Per-step ledger with harness-assigned ids B1, B2, ..."""

    def __init__(self) -> None:
        self.issues: dict[str, Issue] = {}
        self.minor: list[str] = []
        self._counter = 0

    def open_issues(self) -> list[Issue]:
        return [issue for issue in self.issues.values() if issue.status == "OPEN"]

    def add_blocking(self, items: list[str], attempt: int) -> list[Issue]:
        """Open blocking items, retaining only harness ids already known to this ledger."""
        opened = []
        for item in items:
            claimed_id = item.split("|", 1)[0].strip().upper()
            if claimed_id in self.issues:
                issue = self.issues[claimed_id]
                issue.status = "OPEN"
                issue.text = item
            else:
                self._counter += 1
                issue = Issue(f"B{self._counter}", item, attempt)
                self.issues[issue.harness_id] = issue
            opened.append(issue)
        return opened

    def add_minor(self, items: list[str]) -> None:
        for item in items:
            if item not in self.minor:
                self.minor.append(item)

    def apply_previous(self, previous: list[tuple[str, str, str]]) -> list[str]:
        """Update from PREVIOUS lines. Returns ids the auditor did not mention."""
        mentioned = set()
        for auditor_id, status, _evidence in previous:
            issue = self.issues.get(auditor_id.strip().upper())
            if issue is None:
                continue
            mentioned.add(issue.harness_id)
            if status == "RESOLVED":
                issue.status = "RESOLVED"
            else:
                issue.status = "OPEN"
                issue.open_rounds += 1
        unmentioned = [issue.harness_id for issue in self.open_issues() if issue.harness_id not in mentioned]
        for harness_id in unmentioned:
            self.issues[harness_id].open_rounds += 1
        return unmentioned

    def close_all(self) -> None:
        for issue in self.issues.values():
            issue.status = "RESOLVED"

    def to_dict(self) -> dict:
        return {
            "issues": [issue.__dict__.copy() for issue in self.issues.values()],
            "minor": list(self.minor),
        }
