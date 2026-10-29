import re
import time
from dataclasses import asdict, dataclass

from .candidate import Candidate
from .llm import LLMError
from .log import RunLog
from .sandbox import run_snippet
from .verify import verify

# a block opens with ```run or ```submit and closes with a line holding only ```
BLOCK = re.compile(r"^```(run|submit)[ \t]*\r?\n(.*?)^```[ \t]*$", re.S | re.M)
OTHER_FENCE = re.compile(r"^```(\w+)", re.M)


def parse_reply(text):
    """Return (kind, code, n_blocks) for the first run/submit block, or None."""
    blocks = BLOCK.findall(text)
    if not blocks:
        return None
    kind, code = blocks[0]
    return kind, code, len(blocks)


@dataclass
class AgentResult:
    solved: bool  # every check passed, including the reference
    structural_solved: bool  # at some submission every check except the reference passed
    attempts: int
    turns: int
    reason: str
    run_dir: str
    n_run_blocks: int = 0
    n_no_block: int = 0
    first_structural_attempt: int = 0
    first_solved_attempt: int = 0


def run_agent(task, llm, runs_dir="runs", echo=None):
    """Let llm work on the task until a submission passes or a budget runs out."""
    log = RunLog(runs_dir, task, llm)
    say = echo or (lambda *a: None)
    messages = [{"role": "user", "text": "Go ahead."}]
    res = AgentResult(False, False, 0, 0, "turn limit reached", str(log.dir))

    for turn in range(1, task.max_turns + 1):
        res.turns = turn
        t0 = time.time()
        try:
            reply = llm.complete(task.statement, messages)
        except LLMError as exc:
            res.reason = f"llm error: {exc}"
            log.say("error", res.reason)
            log.event(turn=turn, kind="llm_error")
            break
        llm_s = time.time() - t0
        messages.append({"role": "assistant", "text": reply})
        log.say(f"agent (turn {turn})", reply)

        parsed = parse_reply(reply)
        if parsed is None:
            res.n_no_block += 1
            other = OTHER_FENCE.search(reply)
            answer = "No `run` or `submit` block found. Reply with exactly one."
            if other:
                answer += f" (I saw a ```{other.group(1)} block; the tag must be run or submit.)"
            log.event(turn=turn, kind="no_block", llm_s=round(llm_s, 2), reply_chars=len(reply))
            messages.append({"role": "user", "text": answer})
            log.say("harness", answer)
            continue

        kind, code, n_blocks = parsed
        note = "\n(Only the first block was used.)" if n_blocks > 1 else ""

        if kind == "run":
            res.n_run_blocks += 1
            t1 = time.time()
            result = run_snippet(code, timeout=task.run_timeout)
            answer = "output:\n" + (result.output or "(nothing printed)") + note
            log.event(
                turn=turn,
                kind="run",
                llm_s=round(llm_s, 2),
                run_s=round(time.time() - t1, 2),
                timed_out=result.timed_out,
                code_chars=len(code),
                output_chars=len(result.output),
            )
            messages.append({"role": "user", "text": answer})
            log.say("tool", answer)
            continue

        res.attempts += 1
        path = log.save_code(res.attempts, code)
        candidate = Candidate(path=str(path), isolated=True, timeout=task.eval_timeout)
        t1 = time.time()
        report = verify(candidate, suite=task.suite)
        log.record(res.attempts, code, report)
        log.event(turn=turn, kind="submit", attempt=res.attempts, llm_s=round(llm_s, 2),
                  verify_s=round(time.time() - t1, 2), passed=report.passed,
                  structural_ok=report.structural_ok, code_chars=len(code))  # fmt: skip
        say(f"turn {turn}: submission {res.attempts}: {'pass' if report.passed else 'fail'}")
        answer = report.feedback(res.attempts, task.max_attempts) + note
        log.say("verifier", answer)

        if report.structural_ok and not res.structural_solved:
            res.structural_solved, res.first_structural_attempt = True, res.attempts
        if report.passed:
            res.solved, res.first_solved_attempt = True, res.attempts
            res.reason = "all checks passed"
            break
        if res.attempts >= task.max_attempts:
            res.reason = "out of submissions"
            break
        messages.append({"role": "user", "text": answer})

    log.finish(asdict(res))
    return res
