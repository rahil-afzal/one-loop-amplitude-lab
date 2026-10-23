# Contributing

```powershell
python -m pip install -e ".[dev]"
pre-commit install        # optional: ruff + private-key detection
ruff check . ; ruff format .
python -m pytest --cov
```

Rules that keep the validation meaningful:

1. **No check without evidence kind.** Register checks with `@register(name, evidence, description)`
   and pick the weakest honest label.
2. **Every new check needs a negative control** in `tests/test_validation_suite.py`
   (sabotage the physics, assert the check fails).
3. **Read constants at call time** (`conventions.X`, not `from conventions import X`) so
   negative controls can reach them.
4. **Physics core stays LLM-free.** `amplitude_lab.assistant` may import it, never the reverse.
5. **Offline by default.** Tests must pass with no API key and no network. Real network
   tests are marked `gemini` and opt-in.
6. **No credentials in the repo.** Only `.env.example` placeholders.
7. Do not widen the supported scope silently: update `LIMITATIONS` in `reporting.py`
   and `docs/ROADMAP.md` in the same change.
