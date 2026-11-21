# Team and handover record

## Team

| Member | Preferred identifier | Agreed responsibilities | Notes |
|---|---|---|---|
| Member 1 | To be completed | To be agreed | |
| Member 2 | To be completed | To be agreed | |

## Working agreement

Record communication channel, review expectations, task tracking method, and how disagreements or missed handoffs will be handled. These are team decisions, not assessment requirements unless confirmed by the brief.

## Handover log

| Date | Contributor | Completed | Decisions / evidence | Open items and next owner |
|---|---|---|---|---|
| 2026-11-21 | Project scaffold | Created initial folders and documentation only | Official brief was not present in project directory | Add/locate brief; verify assessment requirements; complete team details |


# Team Collaboration – Exercise 2 (CNN)

## 1. Project Overview

This repository contains the PUSL3123 AI and Machine Learning Assessment 1 project.

The project is developed collaboratively by two team members.

| Exercise | Responsibility |
|---|---|
| Exercise 1 – MLP | Member 1 |
| Exercise 2 – CNN | Member 2 |
| Exercise 3 – Regularisation | To be confirmed |
| Exercise 4 – Width and Depth | To be confirmed |

## 2. GitHub Repository

Repository: https://github.com/kavikaran-marazin/PUSL3123-AI-ML-Assessment-1

Both members must work within the same repository.

## 3. Getting Started

Clone the repository:

```bash
git clone https://github.com/kavikaran-marazin/PUSL3123-AI-ML-Assessment-1.git
```

Navigate into the project:

```bash
cd PUSL3123-AI-ML-Assessment-1
```

Switch to the development branch:

```bash
git switch develop
git pull origin develop
```

## 4. Create a Separate Feature Branch

For Exercise 2, create:

```bash
git switch -c feature/exercise-2-cnn
```

Do not directly modify the `main` or `develop` branches.

## 5. AI-Assisted Development Instructions

Before starting implementation, open the project using VS Code or another AI coding assistant.

The AI assistant must first read:

- `.ai/AI_INSTRUCTIONS.md`
- `.ai/PROJECT_CONTEXT.md`
- `.ai/EXPERIMENT_RULES.md`
- `.ai/CODING_STANDARDS.md`
- `docs/assessment-requirements.md`

Also review the official assessment brief.

Use the Exercise 1 MLP notebook as a reference for dataset preparation, preprocessing, experiment organisation and evaluation.

## 6. Exercise 2 – CNN Requirements

The responsible member must:

1. Implement a CNN architecture with exactly three trainable layers, following the assessment brief.
2. Use the same CIFAR-10 dataset.
3. Preserve the existing 45,000 training / 5,000 validation split (seed 42).
4. Keep the original 10,000 test images separate until final evaluation.
5. Use comparable training settings to Exercise 1.
6. Record training and validation loss and accuracy.
7. Evaluate final test performance.
8. Record trainable parameters.
9. Generate and save experiment graphs.
10. Compare CNN and MLP performance.

## 7. Expected Deliverables

Save the work in the following locations:

```text
notebooks/
    exercise_2_cnn.ipynb

experiments/
    plots/
        cnn_loss.png
        cnn_accuracy.png

    results/
        cnn_results.json
```

Additional source code should follow the existing `src/` project structure.

## 8. Git Workflow

After completing and verifying the experiment:

```bash
git add .
git commit -m "Complete Exercise 2 CNN experiment"
git push -u origin feature/exercise-2-cnn
```

Then create a Pull Request on GitHub:

- Source: `feature/exercise-2-cnn`
- Target: `develop`

The other team member will review the changes before merging.

## 9. Important Collaboration Rules

- Do not modify or overwrite Exercise 1 files.
- Maintain the same dataset split and experiment conditions.
- Follow the existing project coding standards.
- Keep experiment results reproducible.
- Do not commit virtual environments or downloaded datasets.
- Understand and verify AI-generated code before using it.
- Follow the assessment's AI-use policy and the module leader's guidance.