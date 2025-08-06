# Health Data Analytics Project 2024

## Overview

This repository contains the complete implementation of solutions for the Health Data Analytics Project 2024, including data processing scripts, SQL queries, Python analysis code, and Jupyter notebooks with comprehensive results.

The project enables full reproducibility of our analytical findings and methodologies.

## Repository Structure

The repository is organized into two main components: database creation and data analysis.

### Database Setup

The [sqlite-scripts](./sqlite-scripts) directory contains Python scripts for creating and normalizing the database from raw CSV data files.

### Data Analysis

The [data-analysis](./data-analysis) directory contains our implementation and results for three distinct data modeling tasks:

1. **Inferential Statistics**: Statistical analysis of treatment outcomes and healthcare utilization patterns
2. **Supervised Learning**: Predictive modeling for patient risk assessment and readmission prevention
3. **Unsupervised Learning**: Patient segmentation and healthcare service optimization

### Notebooks

The [notebooks](./notebooks) directory contains Jupyter notebooks with comprehensive summaries and visualizations for each project task.

## Getting Started

1. Clone this repository
2. Follow the setup instructions in each subdirectory
3. Ensure you have the required CSV data files in the `raw-data` directory
4. Run the database initialization scripts first, then proceed with the analysis

## Requirements

- Python 3.8+
- SQLite3
- Required Python packages (see individual `requirements.txt` files in each directory)