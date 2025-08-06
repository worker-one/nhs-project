# Data Analysis

## Overview

This directory contains the implementation and results for three comprehensive data modeling tasks:

### Task 1: Inferential Statistics
Statistical analysis of treatment outcomes and healthcare utilization patterns, including hypothesis testing and confidence interval estimation.

### Task 2: Supervised Learning
Predictive modeling for patient risk assessment and readmission prevention using machine learning algorithms.

### Task 3: Unsupervised Learning
Patient segmentation and healthcare service optimization through clustering and dimensionality reduction techniques.

## Setup and Execution

### Prerequisites

Ensure the database has been created by running the scripts in the `sqlite-scripts` directory first.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Analysis

Navigate to the specific solution directory and execute the main script:

```bash
cd solution-1
python main.py
```

Repeat for `solution-2` and `solution-3` directories.

## Output

Each solution generates comprehensive results in its respective `assets/` directory:

- **HTML Reports**: Detailed analysis with visualizations
- **Plots**: Statistical charts and machine learning model performance graphs
- **Data Tables**: Summary statistics and model evaluation metrics
- **Console Logs**: Real-time processing information and results

## Results Structure

```
solution-X/
├── assets/
│   ├── report.html
│   ├── plots/
│   └── tables/
├── main.py
└── requirements.txt
```