# Coding standards

These are recommended project conventions, not official assessment requirements. Adapt them if the brief or module guidance specifies otherwise.

- Prefer clear, descriptive names and small functions with one responsibility.
- Keep reusable code in `src/`; use notebooks for exploratory or explanatory work rather than duplicated core logic.
- Make randomness, configuration, data handling, and device selection explicit when implementation begins.
- Avoid hidden state and hard-coded machine-specific paths.
- Add concise docstrings or comments where intent or non-obvious behavior needs explanation.
- Keep dependencies minimal and record selected versions in `requirements.txt` only after verification.
- Do not include credentials, private data, or machine-specific environment files in commits.
