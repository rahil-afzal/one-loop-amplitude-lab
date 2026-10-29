import re
import threading
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


def _format_elapsed(seconds):
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _stage_message(stage, index, total):
    return (
        f"Stage {index + 1}/{total}: {stage['name']}\n\n"
        f"{stage['instruction']}\n\n"
        f"Reply with exactly one `{stage['expected_kind']}` block."
    )


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
    elapsed_seconds: float = 0.0


def run_agent(task, llm, runs_dir="runs", echo=None):
    """Let llm work on the task until a submission passes or a budget runs out."""
    log = RunLog(runs_dir, task, llm)
    say = echo or (lambda *a: None)
    started_at = time.monotonic()

    def progress(message):
        say(f"{message} (elapsed: {_format_elapsed(time.monotonic() - started_at)})")

    stages = getattr(task, "stages", [])
    stage_index = 0
    if stages:
        initial_message = _stage_message(stages[stage_index], stage_index, len(stages))
    else:
        initial_message = "Go ahead."
    messages = [{"role": "user", "text": initial_message}]
    res = AgentResult(False, False, 0, 0, "turn limit reached", str(log.dir))

    for turn in range(1, task.max_turns + 1):
        res.turns = turn
        if stages:
            stage = stages[stage_index]
            progress(f"Stage {stage_index + 1}/{len(stages)}: {stage['name']}")
        t0 = time.monotonic()
        wait_done = threading.Event()

        def report_wait(done=wait_done, current_turn=turn):
            while not done.wait(10):
                progress(
                    f"Still waiting for the model response "
                    f"(turn {current_turn}/{task.max_turns})"
                )

        progress(f"Sending prompt to model (turn {turn}/{task.max_turns})")
        wait_reporter = threading.Thread(target=report_wait, daemon=True)
        wait_reporter.start()
        try:
            reply = llm.complete(task.statement, messages)
        except LLMError as exc:
            llm_s = time.monotonic() - t0
            res.reason = f"llm error: {exc}"
            progress(f"Model request failed after {_format_elapsed(llm_s)}")
            log.say("error", res.reason)
            log.event(turn=turn, kind="llm_error", llm_s=round(llm_s, 2))
            break
        finally:
            wait_done.set()
            wait_reporter.join()
        llm_s = time.monotonic() - t0
        progress(f"Model response received for turn {turn} after {_format_elapsed(llm_s)}")
        messages.append({"role": "assistant", "text": reply})
        log.say(f"agent (turn {turn})", reply)

        parsed = parse_reply(reply)
        if parsed is None:
            res.n_no_block += 1
            other = OTHER_FENCE.search(reply)
            answer = "No `run` or `submit` block found. Reply with exactly one."
            if stages:
                answer = (
                    f"Stage {stage_index + 1}/{len(stages)} ({stage['name']}): "
                    f"reply with exactly one `{stage['expected_kind']}` block."
                )
            if other:
                answer += f" (I saw a ```{other.group(1)} block; the tag must be run or submit.)"
            log.event(turn=turn, kind="no_block", llm_s=round(llm_s, 2), reply_chars=len(reply))
            messages.append({"role": "user", "text": answer})
            log.say("harness", answer)
            continue

        kind, code, n_blocks = parsed
        note = "\n(Only the first block was used.)" if n_blocks > 1 else ""
        if stages and kind != stage["expected_kind"]:
            answer = (
                f"Stay on stage {stage_index + 1}/{len(stages)} ({stage['name']}). "
                f"Reply with exactly one `{stage['expected_kind']}` block for this stage."
            )
            log.event(
                turn=turn,
                kind="stage_mismatch",
                stage=stage_index + 1,
                expected=stage["expected_kind"],
                received=kind,
                llm_s=round(llm_s, 2),
            )
            messages.append({"role": "user", "text": answer})
            log.say("harness", answer)
            continue

        if kind == "run":
            res.n_run_blocks += 1
            t1 = time.monotonic()
            progress(f"Executing code requested by the model (turn {turn})")
            result = run_snippet(code, timeout=task.run_timeout)
            run_s = time.monotonic() - t1
            progress(f"Code execution finished after {_format_elapsed(run_s)}")
            answer = "output:\n" + (result.output or "(nothing printed)") + note
            log.event(
                turn=turn,
                kind="run",
                stage=stage_index + 1 if stages else None,
                llm_s=round(llm_s, 2),
                run_s=round(run_s, 2),
                timed_out=result.timed_out,
                returncode=result.returncode,
                code_chars=len(code),
                output_chars=len(result.output),
            )
            messages.append({"role": "user", "text": answer})
            log.say("tool", answer)
            if stages and result.returncode != 0:
                retry_message = (
                    f"Stage {stage_index + 1}/{len(stages)} ({stage['name']}) did not complete "
                    f"successfully (exit code {result.returncode}). Correct the calculation and "
                    f"retry with one `{stage['expected_kind']}` block."
                )
                messages.append({"role": "user", "text": retry_message})
                log.say("harness", retry_message)
            elif stages and stage_index < len(stages) - 1:
                stage_index += 1
                next_stage = stages[stage_index]
                stage_prompt = _stage_message(next_stage, stage_index, len(stages))
                messages.append({"role": "user", "text": stage_prompt})
                log.say("harness", stage_prompt)
                progress(f"Moving to stage {stage_index + 1}/{len(stages)}: {next_stage['name']}")
            elif stages:
                final_reminder = (
                    f"Stage {stage_index + 1}/{len(stages)} ({stage['name']}): "
                    "now provide the complete candidate in one `submit` block."
                )
                messages.append({"role": "user", "text": final_reminder})
                log.say("harness", final_reminder)
            continue

        res.attempts += 1
        path = log.save_code(res.attempts, code)
        candidate = Candidate(path=str(path), isolated=True, timeout=task.eval_timeout)
        t1 = time.monotonic()
        progress(f"Checking submission {res.attempts} (turn {turn})")
        report = verify(candidate, suite=task.suite)
        verify_s = time.monotonic() - t1
        log.record(res.attempts, code, report)
        log.event(turn=turn, kind="submit", attempt=res.attempts,
                  stage=stage_index + 1 if stages else None, llm_s=round(llm_s, 2),
                  verify_s=round(verify_s, 2), passed=report.passed,
                  structural_ok=report.structural_ok, code_chars=len(code))  # fmt: skip
        progress(
            f"Submission {res.attempts} {'passed' if report.passed else 'failed'} "
            f"after {_format_elapsed(verify_s)}"
        )
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

    res.elapsed_seconds = time.monotonic() - started_at
    log.finish(asdict(res))
    return res
