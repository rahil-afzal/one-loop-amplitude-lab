# Contributing

```powershell
python -m pip install -e ".[dev]"
pre-commit install        # optional: ruff + private-key detection
ruff check . ; ruff format .
python -m pytest --cov
```

Rules:

1. **Offline by default.** Workflow tests use `ScriptedLLM`; they must pass with no API key
   or network access.
2. **Parser security is load-bearing.** Add unsafe-expression, resource-limit, and accepted-
   expression coverage in `tests/test_safe_expr.py`.
3. **The verifier remains repository-referenced.** It checks only the saved final block
   against retained amplitude/physics implementations; document finite-sample limits.
4. **Physics core stays LLM-free.** Workflow modules may import physics, but the reverse
   dependency must remain absent.
5. **No credentials in the repository.** Only `.env.example` placeholders.
6. **Preserve the task resource.** Changes to task packaging or loader behavior require
   wheel/task smoke coverage.
7. **Run the retained checks:** targeted workflow/physics tests, full pytest, and Ruff.
