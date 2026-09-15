# Credit Risk Model: Survival Analysis & Default Prediction

## Project Overview
A discrete-time survival framework for mortgage default, built on the Freddie Mac Single-Family Loan-Level dataset. The project has three components: Kaplan-Meier survival analysis machine learning (LightGBM), and Bayesian/Monte Carlo simulation to tra .

## Data
The project uses the **Freddie Mac Single-Family Loan-Level Dataset**, which covers
fixed-rate mortgages originated from 1999 onward that Freddie Mac either purchased
or used to back mortgage-backed securities. The dataset has two components:

- **Origination file** — one row per loan, capturing underwriting characteristics
  known at the time the loan was made: credit score, LTV, CLTV, DTI, loan purpose,
  occupancy status, property type, channel, first-time homebuyer flag, original
  interest rate, original UPB, and loan term.
- **Performance file** — one row per loan per month, tracking the loan's life:
  current UPB, current interest rate, loan age, delinquency status, modification
  flag, borrower assistance plan, estimated LTV, actual loss, and a zero-balance
  code recording how and when the loan terminated.

The full dataset covers over 50 million loans and several billion monthly
observations. The monthly panel structure is what makes it suitable for survival
modeling

## Data Storage
The data for this project comes from the Freddie Mac Single Family Loan-level dataset. This dataset contains loans originated from 1999 that were sold to Freddie Mac or back Freddie Mac mortgage backed securities. Because it contains over 50 million loan records and their monthly performance the dataset is massive. This project utilizes DuckDB to read the compressed parquet files, process the SQL query, and returns the extracted data.

## Data Preparation & Feature Engineering

1. **Censoring:** In survival analysis, "censoring" means that we don't know the true survival time for that observation (loan). They could have paid off the mortgage early or refinanced. Either way we must censor that mortgage because we don't know if the mortgage resulted in a default. We right-censored loans that had terminated in the previous month.
2. **Forward-Looking Windows:** The analysis 



A LightGBM model was utilized to predict loan defaults.

