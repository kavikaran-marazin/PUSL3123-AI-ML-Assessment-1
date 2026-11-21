# Assessment 1 – Requirements and Traceability

## 1. Source of Truth

**Official source:** PUSL3123 – AI and Machine Learning, Assessment Brief 2026–2027.

**Source file:** `docs/reference/PUSL3123_Assessment_Brief_26-27.pdf`

**Authority:** The official assessment brief is the primary authority for all assessment requirements.

All implementation decisions, experiments, evaluations and submission materials must follow the requirements in this document.

Distinguish between:
- **Official requirement:** Explicitly stated in the assessment brief.
- **Recommended practice:** A suggested approach to improve project quality.
- **Team decision:** An approach agreed by the group.
- **Unverified:** A detail that requires confirmation.

Do not treat recommendations as mandatory assessment requirements.

---

## 2. Assessment Overview

| Item | Details |
|---|---|
| Module | PUSL3123 – AI and Machine Learning |
| Assessment | Assessment 1 – Set Exercises |
| Assessment weighting | 50% |
| Dataset | CIFAR-10 |
| Framework | PyTorch |
| Total exercise marks | 50 |
| Number of exercises | 4 |

**Official source:** Assessment Brief, pages 1–5.

## 3. Dataset Requirements

The assessment requires the CIFAR-10 image classification dataset, available through `torchvision.datasets.CIFAR10`.

Dataset characteristics:

- 60,000 colour RGB images.
- Image dimensions: 32 × 32 pixels.
- 10 classes:
  - Aeroplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

Official dataset partitions:

| Partition | Number of images |
|---|---:|
| Training | 50,000 |
| Test | 10,000 |
| Total | 60,000 |

**Critical restriction:** The original test partition must remain separate. It must not be used for training, hyperparameter tuning or model selection.

**Source:** Assessment Brief, page 5.

### Recommended implementation practice

The group may split the original 50,000 training images into training and validation subsets.

The same training and validation partitions should be reused across the experiments where required by the assessment.

The exact validation split ratio is a team decision, not a specified value in the brief.

---

## 4. Exercise 1 – Multi-Layer Perceptron Classifier

**Weight: 13 marks**

### Official requirements

1. Implement an MLP containing three trainable layers.
2. Select suitable activation functions, a loss function and an optimisation algorithm.
3. Implement data preprocessing, training and evaluation using PyTorch.
4. Evaluate and report training and test performance.
5. Present training and validation loss plots.

**Source:** Assessment Brief, page 5.

### Planned artifacts and evidence

- `notebooks/Exercise_1_MLP.ipynb`
- MLP architecture and configuration.
- Dataset preprocessing and training implementation.
- Training and test performance metrics.
- Training and validation loss plot.
- Explanation of the selected activation, loss and optimiser.

---

## 5. Exercise 2 – Convolutional Neural Network Classifier

**Weight: 10 marks**

### Official requirements

1. Develop a CNN with three trainable layers. Pooling and activation operations may be included as required.
2. Train and evaluate the CNN using the same dataset partitions as Exercise 1.
3. Use comparable training settings wherever practical.
4. Compare MLP and CNN using:
   - Classification accuracy.
   - Training and validation loss.
   - Model parameter counts.
5. Discuss performance differences and explain how convolutional operations influence image feature learning.

**Source:** Assessment Brief, page 5.

### Planned artifacts and evidence

- `notebooks/Exercise_2_CNN.ipynb`
- CNN architecture and configuration.
- Training and evaluation results.
- Trainable parameter counts.
- MLP versus CNN comparison.
- Discussion of convolutional feature learning.

---

## 6. Exercise 3 – Regularisation and Generalisation

**Weight: 17 marks**

### Official requirements

Using the CNN developed in Exercise 2 as the baseline, independently implement and evaluate:

1. **Dropout:** Incorporate dropout into an appropriate part of the CNN.
2. **Data augmentation:** Apply suitable training-image transformations, such as small rotations or translations.

Each technique must initially be evaluated independently against the unchanged CNN baseline.

Additional requirements:

1. Use the same training and validation partitions across all experiments.
2. Keep training conditions consistent wherever possible.
3. Compare training accuracy, test accuracy and the difference between them.
4. Present results in a comparative table or graph.
5. Discuss which techniques improve generalisation, which have limited benefit and possible reasons.

**Source:** Assessment Brief, pages 5–6.

### Planned experiments

| Experiment | Description |
|---|---|
| Baseline | Unchanged CNN from Exercise 2 |
| Dropout | Baseline CNN with dropout |
| Data augmentation | Baseline CNN with training-image augmentation |

### Planned artifacts and evidence

- `notebooks/Exercise_3_Regularisation.ipynb`
- Experiment configurations.
- Comparative results table or graph.
- Training accuracy and test accuracy.
- Difference between training and test accuracy.
- Critical discussion of generalisation.

---

## 7. Exercise 4 – Effect of Model Width and Depth

**Weight: 10 marks**

### Official requirements

Use a CNN architecture to investigate model width and depth.

1. Evaluate three network widths by changing the number of kernels in each hidden layer in increasing order for the architectures.
2. Evaluate three network depths using different depth levels.
3. Evaluate the performance of all six width and depth variations.
4. Keep the dataset split and remaining training conditions consistent.
5. Record:
   - Validation accuracy.
   - Training loss.
   - Number of trainable model parameters.
6. Visualise results using appropriate graphs, such as validation accuracy against selected model widths and depths.
7. Discuss the relationship between model complexity, accuracy, training behaviour and potential overfitting.

**Source:** Assessment Brief, page 6.

### Planned artifacts and evidence

- `notebooks/Exercise_4_Width_Depth.ipynb`
- Three width variations.
- Three depth variations.
- Six experiment results.
- Validation accuracy comparisons.
- Training loss comparisons.
- Trainable parameter counts.
- Appropriate graphs and critical discussion.

---

## 8. Submission Requirements

The group must submit two components.

### 8.1 Experimental Report

**Format:** PDF  
**Word limit:** 3,000 words.

The report must:

- Cover all four exercises in the order presented in the brief.
- Clearly describe model architectures and experimental configurations.
- Present training and evaluation results using suitable tables, graphs and metrics.
- Include comparative analysis of models, techniques and architectural variations.
- Include critical discussion of observations and factors influencing performance.
- Include a clearly identified Individual Contributions section at the start.
- Include a conclusion and relevant references.

The word limit excludes references and the individual contributions section.

### 8.2 Source Code

**Format:** ZIP containing Jupyter Notebooks.

The submission must contain separate notebooks for all four exercises, appropriately named with their exercise numbers.

Code must include:

- Dataset preparation.
- Model implementation.
- Training.
- Evaluation.
- Clear organisation and appropriate comments.
- Executable notebooks.
- Necessary dependencies.
- Instructions to reproduce reported results.

The code and report must be consistent.

Both components must be uploaded to the designated DLE submission area.

**Source:** Assessment Brief, pages 6–7.

---

## 9. Generative AI Usage Policy

**Official classification:** Assisted Work.

The assessment brief explicitly identifies the following acceptable AI uses:

| Category | Description |
|---|---|
| A1 | Idea Generation and Problem Exploration |
| A2 | Planning and Structuring Reports or Projects |
| A4 | Language Refinement |

Requirements:

- AI use must be declared within the documentation.
- A signed Generative AI Declaration must be included as an appendix.
- Students remain responsible for checking AI outputs.
- Students must understand and be able to explain anything they submit.

The declaration form lists additional categories, including coding assistance. However, these additional categories are not automatically authorised by the assessment's stated acceptable-use list.

If the group is unsure whether a particular AI use is permitted, ask the module leader.

**Source:** Assessment Brief, pages 7 and 14.

---

## 10. Code Quality and Version Control

The brief identifies four criteria for good coding practice:

1. Functionality.
2. Efficiency.
3. Readability.
4. Documentation.

Additional official guidance:

- Code must compile or run correctly.
- Necessary dependencies must be included.
- The working process should be visible.
- Incorrect attempts should not automatically be deleted; documented attempts may help demonstrate the development process.
- Code obtained from external sources, including AI-generated code, must be referenced.
- GitHub is recommended by the university for version control.

**Source:** Assessment Brief, pages 3–4.

### Recommended project practices

- Use Git branches and meaningful commits.
- Keep experiment configurations and results organised.
- Maintain reproducible notebooks.
- Document significant implementation decisions.
- Track actual contributions from both group members.

These practices support the assessment but are not additional marking requirements unless specified elsewhere in the brief.

---

## 11. Traceability Matrix

Use this table to connect each official requirement to its implementation and evidence.

| ID | Requirement | Official location | Planned evidence | Status |
|---|---|---|---|---|
| DATA-01 | Use CIFAR-10 | Page 5 | Dataset preparation | Pending |
| DATA-02 | Preserve original test partition | Page 5 | Dataset split verification | Pending |
| EX1-01 | Three trainable MLP layers | Page 5 | Exercise 1 notebook | Pending |
| EX1-02 | Preprocessing, training and evaluation | Page 5 | Exercise 1 notebook | Pending |
| EX1-03 | Training and test performance | Page 5 | Results and metrics | Pending |
| EX1-04 | Training and validation loss plots | Page 5 | Loss graph | Pending |
| EX2-01 | Three trainable CNN layers | Page 5 | Exercise 2 notebook | Pending |
| EX2-02 | Same dataset partitions and comparable settings | Page 5 | Experiment configuration | Pending |
| EX2-03 | MLP/CNN comparison | Page 5 | Comparison table and discussion | Pending |
| EX3-01 | Independent dropout and augmentation experiments | Pages 5–6 | Exercise 3 notebook | Pending |
| EX3-02 | Consistent partitions and conditions | Page 6 | Experiment records | Pending |
| EX3-03 | Training/test accuracy comparison | Page 6 | Comparative table or graph | Pending |
| EX4-01 | Three width variations | Page 6 | Exercise 4 notebook | Pending |
| EX4-02 | Three depth variations | Page 6 | Exercise 4 notebook | Pending |
| EX4-03 | Evaluate all six variations | Page 6 | Experiment results | Pending |
| EX4-04 | Record validation accuracy, training loss and parameters | Page 6 | Results table | Pending |
| EX4-05 | Visualise and discuss results | Page 6 | Graphs and analysis | Pending |
| SUB-01 | 3,000-word PDF report | Page 6 | Final report | Pending |
| SUB-02 | Four separate Jupyter Notebooks in ZIP | Page 7 | Source code ZIP | Pending |
| SUB-03 | Include dependencies and reproduction instructions | Page 7 | Requirements and README | Pending |
| AI-01 | Follow Assisted Work policy and declare AI use | Pages 7 and 14 | AI declaration | Pending |

---

## 12. Clarifications and Ambiguities

The brief contains two points that should be checked with the module team:

1. Page 2 refers to COMP2000 (Software Engineering 2) in the overview, while the actual document identifies PUSL3123 – AI and Machine Learning.
2. Page 2 lists the Set Exercises deadline as 7 December 2027 and the Report deadline as 12 January 2027. These dates appear chronologically inconsistent and should be confirmed with the module team or DLE.

Do not silently change these dates or assume a corrected deadline.

---

## 13. Document Maintenance

Update this document only when:

- An official requirement is clarified by teaching staff.
- A team decision affects how a requirement will be implemented.
- Requirement status changes based on verified work.

Keep the distinction between official requirements, recommended practices, team decisions and unverified details.

**Last reviewed:** 2026-11-21  
**Review status:** Transcribed from the available official assessment brief; verify any future clarifications with the module team.