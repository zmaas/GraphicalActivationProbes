# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Setup: `uv pip install -r requirements.txt`
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate`
- Run: `python main.py`
- Test: `pytest`
- Single test: `pytest path/to/test.py::test_function`
- Lint: `ruff check .`
- Type check: `mypy .`

## Code Style Guidelines
- Python: Follow PEP 8 standards
- Use type hints in function signatures
- Imports: standard library, third-party, local (alphabetical in groups)
- Naming: snake_case for functions/variables, CamelCase for classes
- Docstrings: Google style docstrings for functions/classes
- Error handling: Use explicit try/except blocks with specific exceptions
- Log errors properly instead of printing
- Functions should have single responsibility
- Prefer functional approaches for tensor operations