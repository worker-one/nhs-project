# Database Setup Scripts

## Overview

This directory contains Python scripts for creating and normalizing the NHS database from raw CSV data files. The scripts handle data cleaning, validation, and proper database schema creation.

## Prerequisites

### Required Data Files

Place the following CSV files in the [raw-data](../raw-data/) directory:

- `appointments_data.csv`
- `medical_appointments_data.csv`
- `medical_surgeries_data.csv`
- `medical_tests_data.csv`
- `prescription_billing_insurance_data.csv`
- `service_billing_insurance_data.csv`

## Setup and Execution

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Database Initialization

```bash
python init_db.py
```

## Output

Upon successful execution, the script will generate:

- `nhs_db.db` - The normalized SQLite database file
- Detailed processing logs displayed in the console
- Data validation reports and statistics

## Database Schema

The resulting database contains normalized tables with proper relationships, indexes, and constraints to ensure data integrity and optimal query performance.