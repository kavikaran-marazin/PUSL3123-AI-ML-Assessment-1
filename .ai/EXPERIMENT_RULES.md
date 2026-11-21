# Experiment rules

No experiments have been started. The following are recommended practices for later work, subject to the official brief.

- Confirm required comparisons, metrics, data splits, and reporting constraints from the official brief before designing runs.
- Give each run a stable identifier and record its purpose, date, code revision, configuration, seed (if used), data handling, environment, and output paths.
- Preserve raw logs and results; do not silently replace or selectively omit unsuccessful runs.
- Separate measured results from interpretation and label exploratory runs as such.
- Keep large generated artifacts out of Git unless the submission instructions require them; document where required artifacts are stored.
- Do not download CIFAR-10 or launch training until explicitly requested and the brief's data rules are known.
