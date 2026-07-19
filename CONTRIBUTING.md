# Contributing

Bug reports, reproducibility checks, and focused pull requests are welcome.

1. Create a branch from `main`.
2. Install the development environment with `pip install -e ".[all]"`.
3. Add or update tests for semantic or cache changes.
4. Run `pytest` and `python -m py_compile` over modified Python files.
5. Document any new experiment configuration and its expected outputs.

Rule semantics and Chimera target construction are part of the scientific specification. Changes to them should include a truth-table test and a short explanation in the pull request.
