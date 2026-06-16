# Contributing to Life Memorizer

Thanks for your interest in improving Life Memorizer! This guide explains how to set up,
make changes, and submit them.

## Code of conduct

Be respectful and constructive in issues, merge requests, and reviews. Assume good intent.

## Getting started

1. **Fork** the project (or create a branch if you have access):
   <https://gitlab.com/satyam671-group/life-memorizer>
2. **Clone** your fork:
   ```bash
   git clone https://gitlab.com/<your-namespace>/life-memorizer.git
   cd life-memorizer
   ```
3. **Create a dev environment** and install everything:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e ".[media,dev]"
   ```
4. **Verify your setup** runs offline:
   ```bash
   LIFE_MEMORIZER_FAKE_EMBEDDINGS=1 pytest
   ```

## Making a change

1. Create a focused branch:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make your change. Please:
   - Add type annotations; prefer Pydantic models for data.
   - Add or update tests in `tests/` — they must pass **offline** (no API key, no media).
   - Keep functions small and add docstrings.
   - Match the existing code style.
3. Run the checks before committing:
   ```bash
   ruff check .
   pytest
   ```
4. Commit with a clear, conventional message:
   ```bash
   git commit -m "feat: add binary quantization toggle"
   ```
   Prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
5. Push and open a **Merge Request** against `main`:
   ```bash
   git push origin feat/short-description
   ```
   Describe *what* changed and *why*, and link any related issues (e.g. `Closes #12`).

## Guidelines

- **Tests must pass offline**: `LIFE_MEMORIZER_FAKE_EMBEDDINGS=1 pytest`.
- **Never commit secrets.** `.env` is git-ignored; do not hard-code API keys.
- **Don't commit generated data**: local DB (`life_memorizer_db/`), media files (`*.mp4`, `*.wav`) — these are already in `.gitignore`.
- **Keep public APIs stable.** If you must break one, call it out in the MR.
- **Small, reviewable MRs** are easier to merge than large ones.

## Reporting bugs and requesting features

Open an issue: <https://gitlab.com/satyam671-group/life-memorizer/-/issues>

For bugs, include:
- Steps to reproduce
- Expected vs. actual behavior
- OS and Python version
- Whether you used real or stub embeddings (`LIFE_MEMORIZER_FAKE_EMBEDDINGS`)

Thank you for contributing!
