# Credit Risk Model: Survival Analysis & Default Prediction

## Project Overview
A discrete-time survival framework for mortgage default, built on the Freddie Mac Single-Family Loan-Level dataset. The project has three parts: a Kaplan-Meier survival analysis of time-to-default, a LightGBM model predicting the probability of default within a forward-looking 12-month window, and a Monte Carlo simulation that translates segment-level default and loss rates into a portfolio loss distribution and Value at Risk.

## Contents

- [Data](#data)
- [Data Storage](#storage-architecture)
- [Data Preparation & Feature Engineering](#sample-construction)
  - [Loan-level sampling](#loan-level-sampling)
  - [Defining the default event](#defining-the-default-event)
  - [Censoring](#censoring)
  - [Forward-looking prediction windows](#forward-looking-prediction-windows)
  - [Feature engineering](#feature-engineering)
  - [Temporal splits](#temporal-splits)
  - [Downsampling](#downsampling)
- [Macroeconomic covariates](#macroeconomic-covariates)
- [Kaplan-Meier survival analysis](#kaplan-meier-survival-analysis)
- [The 12-month default model](#the-12-month-default-model)
- [Hyperparameter optimization](#hyperparameter-optimization)
- [Isolation Forest: a negative result](#isolation-forest-a-negative-result)
- [Monte Carlo portfolio loss simulation](#monte-carlo-portfolio-loss-simulation)
- [Limitations](#limitations)
- [Reproducing this work](#reproducing-this-work)

---


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

The raw distribution is a set of pipe-delimited text files inside per-quarter zip
archives. Reading those directly is impractical at this scale, so the first step
converts them into a columnar store.

Each quarterly archive is unzipped, its origination and performance files are read
with the pyarrow CSV engine, column names are applied from the header definition
files, a `quarter` column is added, and the result is written to Parquet with ZSTD
compression:

```python
df_quarter = pd.read_csv(z.open(orig_file), sep="|", header=None, engine='pyarrow')
df_quarter.columns = orig_columns
df_quarter['quarter'] = quarter
df_quarter.to_parquet(
    os.path.join(orig_output_dir, f'orig_{quarter}.parquet'),
    compression='zstd', index=False
)
```
## Data Preparation & Feature Engineering


## Sample construction

The heart of the project is one DuckDB query that goes from raw monthly panel to a
model-ready dataset. It runs as a chain of CTEs, each handling one concern. The
sections below walk through them in order.


The resulting ID list is small enough to act as a fast join filter. In the main
query it is joined to the performance file *first*, so the expensive join to
origination data only runs against the 10% subset:

```sql
FROM read_parquet('{PERF_PARQUET_PATH}', union_by_name=true) p
INNER JOIN sampled_ids s ON p."LOAN IDENTIFIER" = s."LOAN IDENTIFIER"
INNER JOIN read_parquet('{ORIG_PARQUET_PATH}', union_by_name=true) o
    ON p."LOAN IDENTIFIER" = o."LOAN IDENTIFIER"
```

### Defining the default event

A loan-month is flagged as a default event under any of three conditions:

```sql
CASE
    WHEN p."CURRENT LOAN DELINQUENCY STATUS" IN ('RA') THEN 1
    WHEN p."CURRENT LOAN DELINQUENCY STATUS" NOT IN ('XX', '00', '01', '02')
         AND p."CURRENT LOAN DELINQUENCY STATUS" ~ '^[0-9]+$'
         AND TRY_CAST(p."CURRENT LOAN DELINQUENCY STATUS" AS INTEGER) >= 3 THEN 1
    WHEN p."ZERO BALANCE CODE" IN ('03', '09') THEN 1
    ELSE 0
END AS default_event
```

In plain terms: 90+ days delinquent, in REO acquisition (`RA`), or terminated
through a short sale / charge-off (`03`) or REO disposition (`09`).


### Censoring

Once a loan defaults or terminates, its subsequent rows are no longer valid
observations — the loan is no longer at risk. A `LAG` window function flags any
month following a termination event, and those rows are dropped:

```sql
censored_data AS (
    SELECT *,
        LAG(CASE WHEN default_event = 1 OR "ZERO BALANCE CODE" IS NOT NULL
                 THEN 1 ELSE 0 END)
        OVER (PARTITION BY "LOAN IDENTIFIER" ORDER BY "PERIOD")
            AS terminated_prev_month
    FROM base_data
)
```

```sql
FROM censored_data
WHERE COALESCE(terminated_prev_month, 0) = 0
```


### Forward-looking prediction windows

The target is not "is this loan in default now" but "will this loan default within
the next 12 months." Three window functions construct it, all looking strictly
forward from the current row:

```sql
MAX(default_event) OVER (
    PARTITION BY "LOAN IDENTIFIER"
    ORDER BY "PERIOD"
    ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING
) AS default_in_next_12m,

MAX(CASE WHEN "ZERO BALANCE CODE" IS NOT NULL THEN 1 ELSE 0 END) OVER (
    PARTITION BY "LOAN IDENTIFIER"
    ORDER BY "PERIOD"
    ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING
) AS terminates_in_next_12m,

COUNT(*) OVER (
    PARTITION BY "LOAN IDENTIFIER"
    ORDER BY "PERIOD"
    ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING
) AS future_months_12m
```

`ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING` is the key clause: it excludes the
current row, so a loan already in default does not trivially predict its own
target.

These three windows combine into the label:

```sql
CASE
    WHEN future_months_12m < 12 THEN NULL
    WHEN terminates_in_next_12m = 1 AND COALESCE(default_in_next_12m, 0) = 0 THEN NULL
    WHEN COALESCE(default_in_next_12m, 0) = 1 THEN 1
    ELSE 0
END AS default_12m
```

Two distinct sources of censoring produce a `NULL` rather than a zero:

1. **Administrative censoring.** Fewer than 12 future months exist in the data, so
   the outcome is unobservable. This affects loans near the end of the observation
   period.
2. **Competing-risk censoring.** The loan terminates within the window for a reason
   other than default — most commonly prepayment. Labeling these as zero would tell
   the model that a loan that refinanced away is evidence of creditworthiness over
   a window it never actually faced.

Rows where the target is `NULL` are excluded at training time rather than imputed.
This is the standard discrete-time survival treatment: an unobserved outcome is not
a negative outcome.

The same structure is built for an 18-month horizon in parallel. That variant is
not documented here.

### Feature engineering

**Sentinel handling.** Freddie Mac encodes missing values as out-of-range sentinels
rather than nulls. Left untreated, a model reads a DTI of 999 as an extreme but
real value. Each is mapped explicitly:

```sql
CASE WHEN CAST(o."CLASSIC FICO" AS VARCHAR) IN ('9999') THEN NULL
     ELSE TRY_CAST(o."CLASSIC FICO" AS INTEGER) END AS CLASSIC_FICO,
CASE WHEN CAST(o."ORIGINAL DEBT-TO-INCOME (DTI) RATIO" AS VARCHAR) IN ('999') THEN NULL
     ELSE TRY_CAST(o."ORIGINAL DEBT-TO-INCOME (DTI) RATIO" AS INTEGER) END AS ORIGINAL_DTI,
```

Continuous fields (`CLASSIC_FICO` 9999; `ORIGINAL_LTV`, `ORIGINAL_CLTV`,
`ORIGINAL_DTI`, `ELTV` 999) become `NULL`, which LightGBM handles natively by
learning a default split direction. Categorical sentinels (`9`, `99`) become an
explicit `'Missing'` category, so that missingness can carry signal if it has any.
`BORROWER ASSISTANCE PLAN` nulls become `'No_Plan'`, since absence there means
something specific.

**Derived features.**

```sql
-- Rate movement since origination
CASE WHEN TRY_CAST(p."CURRENT INTEREST RATE" AS DOUBLE) IS NOT NULL
      AND TRY_CAST(o."ORIGINAL INTEREST RATE" AS DOUBLE) IS NOT NULL
     THEN TRY_CAST(p."CURRENT INTEREST RATE" AS DOUBLE)
        - TRY_CAST(o."ORIGINAL INTEREST RATE" AS DOUBLE)
     ELSE NULL END AS PAYMENT_SHOCK,

-- Origination year, parsed from positions 2-3 of the loan identifier
CASE WHEN CAST(SUBSTRING("LOAN IDENTIFIER", 2, 2) AS INTEGER) = 99 THEN 1999
     ELSE 2000 + CAST(SUBSTRING("LOAN IDENTIFIER", 2, 2) AS INTEGER) END AS ORIG_YEAR,

-- Amortization progress: share of original balance still outstanding
("CURRENT ACTUAL UPB" / NULLIF("ORIGINAL UPB", 0)) * 100 AS PROXY_LTV
```

`PAYMENT_SHOCK` captures rate movement on modified loans. `PROXY_LTV` is a
time-varying measure of how far the loan has amortized, available for every
loan-month; it is not a true LTV, since the denominator is the original balance
rather than a current property value. `ORIG_YEAR` is derived from the loan ID
rather than from a date field, and is used only to assign splits.

### Temporal splits

Splits are assigned by **origination year**, not randomly:

```sql
CASE
    WHEN ORIG_YEAR <= 2015 THEN 'train'
    WHEN ORIG_YEAR BETWEEN 2016 AND 2018 THEN 'val'
    ELSE 'test'
END AS data_split
```

| Split | Origination years | Rationale |
|---|---|---|
| Train | 1999–2015 | Spans the full credit cycle, including the 2008 crisis |
| Validation | 2016–2018 | Post-crisis, pre-pandemic |
| Test | 2019+ | Most recent vintages, including COVID-era performance |

A random split would leak information across time: a model could learn from a
loan's 2020 observations to predict its 2019 ones. Splitting by vintage means the
test set genuinely evaluates performance on loans the model has never seen, in an
economic environment it was not trained on. It also means the test set is
structurally harder, and the reported metrics are conservative relative to what a
random split would produce.

### Downsampling

Defaults are rare. Training on the full monthly panel means the overwhelming
majority of gradient computation is spent on obvious non-defaults. The pipeline
keeps every default and a 10% random sample of non-defaults — **in the training
split only**:

```sql
downsampled_data AS (
    SELECT * FROM clean_data WHERE data_split IN ('val', 'test')
    UNION ALL
    SELECT * FROM clean_data
        WHERE data_split = 'train' AND (default_12m = 1 OR default_18m = 1)
    UNION ALL
    SELECT * FROM clean_data
        WHERE data_split = 'train'
          AND COALESCE(default_12m, 0) = 0
          AND COALESCE(default_18m, 0) = 0
          AND RANDOM() < 0.10
)
```

Validation and test pass through untouched. This is the point that matters: if the
evaluation sets were also downsampled, every reported metric would describe an
artificial population rather than the real portfolio.

The resulting class balance:

| Split | Rows | 12m defaults | 12m default rate |
|---|---:|---:|---:|
| Train | 3,740,535 | 1,469,484 | 39.29% |
| Validation | 19,138,933 | 249,767 | 1.31% |
| Test | 41,548,238 | 260,805 | 0.63% |

The 39% training rate is a direct artifact of the downsampling and the reason the
model's raw output scores are not calibrated probabilities. See
[Limitations](#limitations).

## Macroeconomic covariates

Three series are pulled from FRED and merged onto each loan-month by period:

| Feature | FRED series | Frequency |
|---|---|---|
| `GDP` | `GDP` | Quarterly, interpolated to monthly |
| `UNEMPLOYMENT_RATE` | `UNRATE` | Monthly |
| `HOME_PRICE_INDEX` | `CSUSHPISA` | Monthly |

The pull uses `get_series_first_release()` rather than the current revised series.
This is deliberate: it retrieves the vintage value as originally published, so the
model sees the number that would actually have been available at that point in
time, not a figure revised years later.

GDP is shifted two months forward before interpolation to approximate publication
lag, then linearly interpolated to a monthly frequency.

```python
raw_series = fred_client.get_series_first_release(series_id)
...
df['GDP'] = df['GDP'].shift(2, axis=0, fill_value=pd.NA)
df['GDP'] = df['GDP'].interpolate(method='linear')
```

## Kaplan-Meier survival analysis

Before fitting any model, the survival analysis establishes the baseline shape of
default risk over a loan's life, and tests whether occupancy status alone separates
the curves.

Kaplan-Meier estimates the survival function. It handles right-censoring directly, which matters here
because most loans in the data have not defaulted and are still being observed.

The monthly panel is first collapsed to one row per loan: a duration and a binary
event indicator.

```sql
WITH loan_outcomes AS (
    SELECT
        "LOAN IDENTIFIER",
        CAST(MAX("LOAN AGE") AS INTEGER) AS final_age,
        CAST(MAX(CASE
            WHEN "CURRENT LOAN DELINQUENCY STATUS" >= '03' THEN 1
            WHEN "ZERO BALANCE CODE" IN ('03', '09') THEN 1
            ELSE 0
        END) AS INTEGER) AS event
    FROM read_parquet('{PERF_PARQUET_PATH}')
    GROUP BY "LOAN IDENTIFIER"
)
SELECT
    TRIM(o."OCCUPANCY STATUS") AS OCCUPANCY_STATUS,
    lo.final_age,
    lo.event
FROM loan_outcomes lo
JOIN read_parquet('{ORIG_PARQUET_PATH}') o
  ON lo."LOAN IDENTIFIER" = o."LOAN IDENTIFIER"
WHERE TRIM(o."OCCUPANCY STATUS") IN ('P', 'I')
```

Two groups are compared — owner-occupied primary residences (`P`) against
investment properties (`I`) — fitted with `lifelines` and tested with a log-rank
test:

```python
kmf_primary.fit(df_primary["final_age"], event_observed=df_primary["event"],
                label="Primary Residence (P)")
kmf_investment.fit(df_investment["final_age"], event_observed=df_investment["event"],
                   label="Investment Property (I)")

results = logrank_test(
    df_primary["final_age"], df_investment["final_age"],
    event_observed_A=df_primary["event"], event_observed_B=df_investment["event"]
)
```

The log-rank test asks whether the two survival curves differ more than chance
would explain, across the whole follow-up period rather than at a single horizon.


## The 12-month default model

### Framing

The model is a **discrete-time survival model**, not a conventional binary
classifier on loans. Each row is a loan-month: a loan that survives 60 months
contributes 60 observations, each asking whether default occurs in the following 12
months conditional on surviving to that point. Gradient boosting then estimates
that conditional hazard without any proportional-hazards assumption, and captures
interactions — low FICO combined with high LTV, say — that a Cox model would need
specified by hand.

### Features

Sixteen features, of which five are categorical and handled natively by LightGBM
rather than one-hot encoded:

```python
features = ['CLASSIC_FICO', 'PROXY_LTV', 'ORIGINAL LOAN TERM', 'LOAN AGE',
            'GDP', 'UNEMPLOYMENT_RATE', 'HOME_PRICE_INDEX', 'ORIGINAL_DTI',
            'ORIGINAL_LTV', 'CHANNEL', 'ELTV', 'OCCUPANCY_STATUS',
            'PAYMENT_SHOCK', 'BORROWER_ASSISTANCE_PLAN', 'PROPERTY_TYPE',
            'LOAN_PURPOSE']

categorical_features = ['CHANNEL', 'OCCUPANCY_STATUS', 'BORROWER_ASSISTANCE_PLAN',
                        'PROPERTY_TYPE', 'LOAN_PURPOSE']
```

`LOAN AGE` is the baseline hazard term — it lets the model learn the shape of the
seasoning curve directly from the data.

Loan identifiers, period, and raw balance fields are dropped. So is
`CURRENT LOAN DELINQUENCY STATUS`: it is the variable the target is derived from,
and leaving it in would leak the outcome.

### Training

```python
params = {
    'objective': 'binary',
    'metric': 'average_precision',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 10,
    'min_child_samples': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1
}

model_12m = lgb.train(
    params, lgb_train_12m,
    num_boost_round=1500,
    valid_sets=[lgb_val_12m],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
)
```

**Average precision (PR-AUC) is the early-stopping metric, not ROC-AUC.** With a
0.63% positive rate, ROC-AUC is dominated by the model's ability to rank the vast
negative class and stays high even when the model is nearly useless at the decision
boundary. PR-AUC is sensitive to performance on the positive class, which is what
actually matters.

`min_child_samples=100` prevents leaves forming on tiny subgroups;
`feature_fraction` and `bagging_fraction` at 0.8 add stochastic regularization.
Early stopping on the validation set — a different set of origination vintages —
selected iteration 420 of a possible 1500, in 589 seconds.

### Results

| Metric | Test value |
|---|---|
| PR-AUC (average precision) | 0.1158 |
| ROC-AUC | 0.7997 |
| Test set size | 41,548,238 loan-months |
| Test default rate | 0.63% |

PR-AUC of 0.116 against a 0.63% base rate is roughly an 18x lift over random
ranking. ROC-AUC of 0.80 indicates the model orders risk well across the full
population.

### Feature importance

![SHAP global feature importance, 12-month model](images/shap_importance_12m.png)

SHAP values decompose each prediction into per-feature contributions, giving both
the magnitude and the direction of each feature's effect — closer to a regression
coefficient than a split-count importance.

The macroeconomic features dominate. `HOME_PRICE_INDEX` and `GDP` rank first and
second, ahead of `CLASSIC_FICO` in third. That ordering is consistent with the
credit-risk literature: on a portfolio of already-underwritten loans, the
environment moves default risk more than the underwriting spread does. `LOAN AGE`
and `PROXY_LTV` follow, capturing seasoning and amortization. `PAYMENT_SHOCK`
contributes almost nothing, which follows from the sample — these are fixed-rate
loans, so rates move only for the small modified population.

![SHAP beeswarm, 12-month model](images/shap_beeswarm_12m.png)

The beeswarm shows directionality: high FICO values push predictions down, low
values push them up, and the spread widens at the low end.

### Threshold behavior

![Precision and recall across thresholds](images/precision_recall_vs_threshold.png)

![Precision-recall curve](images/precision_recall_curve.png)

Precision and recall for the 12-month model at selected cutoffs:

| Threshold | Precision | Recall | Loans flagged |
|---:|---:|---:|---:|
| 0.05 | 0.0063 | 1.0000 | 41,526,615 |
| 0.50 | 0.0091 | 0.9413 | 26,912,125 |
| 0.70 | 0.0199 | 0.6040 | 7,920,705 |
| 0.80 | 0.0950 | 0.2668 | 732,160 |
| 0.90 | 0.4223 | 0.1324 | 81,792 |
| 0.95 | 0.4872 | 0.0634 | 33,940 |

The useful range sits at the high end. Below 0.70 the model flags most of the
portfolio; at 0.90 it flags 81,792 loans and is right 42% of the time, against a
0.63% base rate. **These thresholds are specific to this model's uncalibrated score
distribution and do not correspond to probabilities of default** — a score of 0.90
does not mean a 90% chance of default. See [Limitations](#limitations).

## Hyperparameter optimization

An Optuna study ran 100 trials with a TPE sampler, tuning ten parameters against
validation PR-AUC.

```python
params = {
    'num_leaves':        trial.suggest_int('num_leaves', 31, 100),
    'max_depth':         trial.suggest_int('max_depth', 5, 10),
    'min_child_samples': trial.suggest_int('min_child_samples', 100, 500),
    'feature_fraction':  trial.suggest_float('feature_fraction', 0.6, 0.95),
    'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.6, 0.95),
    'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
    'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
    'lambda_l1':         trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
    'lambda_l2':         trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
}
```

Three choices made the search tractable. The validation set was subsampled to about
2.25M rows, keeping **every** positive and 2M sampled negatives — subsampling the
rare class would have made the objective too noisy to optimize against. Datasets
were serialized to LightGBM binary format so four parallel trials could each hold
an isolated copy. A `MedianPruner` terminated trials falling below the running
median after 30 warmup steps.

Best parameters:

```python
best_params = {
    'num_leaves': 40, 'max_depth': 8, 'min_child_samples': 106,
    'feature_fraction': 0.6458636185880158,
    'bagging_fraction': 0.8332057315817041,
    'bagging_freq': 1,
    'learning_rate': 0.06139493559139317,
    'lambda_l1': 1.0818776346214012,
    'lambda_l2': 0.11625491577945968,
    'min_gain_to_split': 0.9868557073620492
}
```

The search moved toward shallower trees (depth 8 vs 10), fewer features per tree
(0.65 vs 0.80), and meaningful L1 regularization — all pointing the same direction:
the baseline was slightly overfitting.

The model was then retrained on the full training set against the full validation
set, stopping at iteration 129.

### Optimized vs baseline

![Original vs optimized model comparison](images/original_vs_optimized.png)

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| PR-AUC | 0.1158 | 0.1220 | +0.0062 |
| ROC-AUC | 0.7997 | 0.7948 | −0.0049 |

A 5.4% relative improvement in PR-AUC, with ROC-AUC slightly worse. That trade is
the intended one — PR-AUC was the optimization target — but the gain is small
enough to be worth stating plainly: 100 trials of tuning bought roughly half a
percentage point of average precision. The pipeline's data-construction decisions
mattered considerably more than the hyperparameters.

The gains concentrate in the high-threshold region. At 0.80 the optimized model's
precision rises from 0.095 to 0.158, flagging half as many loans (370,658 vs
732,160) for a recall cost of four percentage points.

## Isolation Forest: a negative result

An Isolation Forest was tested on the hypothesis that defaults might be detectable
as anomalies, without supervision. It was not.

```python
features = ['CLASSIC_FICO', 'PROXY_LTV', 'ORIGINAL LOAN TERM', 'LOAN AGE',
            'ORIGINAL_DTI', 'ORIGINAL_LTV']

params = {
    'n_estimators': 200,
    'max_samples': 'auto',
    'contamination': contamination_rate,   # 0.0130, the training default rate
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1,
    'random_state': 42
}
```

| Metric | Validation | Test |
|---|---:|---:|
| PR-AUC | 0.0011 | 0.0008 |
| ROC-AUC | 0.4368 | 0.4961 |

A ROC-AUC of 0.4961 is indistinguishable from random. Validation ROC-AUC of 0.437
is worse than random — the anomaly score is weakly *inversely* related to default.
Of 87,335 flagged anomalies, precision on defaults rounds to zero.

![SHAP importance, Isolation Forest](images/shap_importance_isolation_forest.png)

![SHAP beeswarm, Isolation Forest](images/shap_beeswarm_isolation_forest.png)

The result is not surprising in hindsight, and the reason is worth stating.
Isolation Forest finds points that are *unusual* in feature space. Defaulting
mortgages are not unusual — a 680 FICO, 90% LTV, 38% DTI loan is an entirely
ordinary loan that happens to default more often than average. Statistically
unusual loans are as likely to be exceptionally safe as exceptionally risky.
Default is a conditional-probability problem, not an outlier problem, and
unsupervised anomaly detection cannot substitute for a label.

This model is documented because the negative result is informative, not because it
is usable. It should not be deployed or extended.

## Monte Carlo portfolio loss simulation

The classification model ranks individual loans. The Monte Carlo simulation answers
a portfolio-level question: what is the distribution of total credit loss over the
next year, and how bad is a bad year?

### Estimating PD and LGD by credit segment

Loans are bucketed by origination FICO, then collapsed to one row per loan and
aggregated:

```sql
CASE
    WHEN TRY_CAST(o."CLASSIC FICO" AS INTEGER) IS NULL THEN 'Missing'
    WHEN TRY_CAST(o."CLASSIC FICO" AS INTEGER) < 620 THEN 'Sub_620'
    WHEN TRY_CAST(o."CLASSIC FICO" AS INTEGER) >= 620
     AND TRY_CAST(o."CLASSIC FICO" AS INTEGER) < 679 THEN '620_to_679'
    WHEN TRY_CAST(o."CLASSIC FICO" AS INTEGER) >= 680
     AND TRY_CAST(o."CLASSIC FICO" AS INTEGER) < 739 THEN '680_to_739'
    ELSE '740_plus'
END AS CREDIT_SCORE_BUCKET
```

```sql
bucketed_outcomes AS (
    SELECT
        "CREDIT_SCORE_BUCKET",
        SUM(ORIGINAL_UPB) AS TOTAL_EXPOSURE,
        SUM(is_defaulted) * 1.0 / COUNT("LOAN IDENTIFIER") AS PD,
        CASE
            WHEN SUM(CASE WHEN is_defaulted = 1 THEN ORIGINAL_UPB ELSE 0 END) > 0
            THEN SUM(estimated_loss_amount) * 1.0
               / SUM(CASE WHEN is_defaulted = 1 THEN ORIGINAL_UPB ELSE 0 END)
            ELSE 0
        END AS LGD
    FROM loan_level_outcomes
    GROUP BY "CREDIT_SCORE_BUCKET"
)
```

`PD` is the observed default frequency within the bucket. `LGD` is realized losses
as a share of the balance at risk, computed only over loans that actually
defaulted. Loss amounts come from the performance file's `ACTUAL LOSS` field on
terminated loans.

### The simulation

Expected loss follows the standard decomposition:

```python
df['Expected_Loss'] = df['TOTAL_EXPOSURE'] * df['PD'] * df['LGD']
```

| Credit bucket | Expected loss |
|---|---:|
| 680–739 | $711,144,700 |
| 620–679 | $606,187,800 |
| 740+ | $510,460,000 |
| Sub-620 | $150,751,600 |

Expected loss is an average, and averages do not size capital. The simulation draws
10,000 scenario years, in each drawing a loss for every bucket from a normal
distribution centered on that bucket's expected loss:

```python
num_simulations = 10000
volatility = 0.25

for i in range(num_simulations):
    scenario_total_loss = 0
    for index, row in df.iterrows():
        expected = row['Expected_Loss']
        simulated_loss = np.random.normal(loc=expected, scale=expected * volatility)
        simulated_loss = max(0, simulated_loss)   # losses cannot be negative
        scenario_total_loss += simulated_loss
    portfolio_losses.append(scenario_total_loss)

VaR_95 = np.percentile(portfolio_losses, 95)
VaR_99 = np.percentile(portfolio_losses, 99)
```

Value at Risk is then read off as a percentile of the simulated distribution: the
99% VaR is the loss level exceeded in only 1% of simulated years.

### Results

![Monte Carlo portfolio loss distribution](images/monte_carlo_loss_distribution.png)

| Measure | Value |
|---|---:|
| Expected loss (mean) | $1,978,477,718 |
| 95% VaR (1-year) | $2,424,061,705 |
| 99% VaR (1-year) | $2,609,291,586 |

The gap between expected loss and 99% VaR — about $631 million — is the
unexpected-loss component, and under a Basel-style framework it is what economic
capital is sized against.

Two caveats belong with these numbers rather than after them. The 25% volatility is
an assumption, not an estimate. And with only four buckets drawn independently,
diversification across them is close to complete, so the distribution is far
narrower than a real mortgage portfolio's would be. Both are expanded below.

## Limitations

These are material and should be read before citing any number above.

### The model's output scores are not probabilities of default

Downsampling produced a 39.29% default rate in training against a 0.63% rate in
test. The model learned a decision function calibrated to the training
distribution, so predicted scores are centered near 0.56 (median 0.59) when the
true rate is under 1%.

This does not affect ranking metrics — ROC-AUC and PR-AUC are invariant to monotone
transformations of the score — but it means:

- A score of 0.90 is **not** a 90% probability of default. It is a high rank.
- The threshold tables above describe this score distribution only, and are not
  transferable to another model or dataset.
- The scores cannot be used directly for expected-loss calculation, pricing, or
  capital purposes without recalibration.

The fix is a standard prior correction or Platt scaling against the true base rate.
Until that is applied, treat the output as an ordinal risk ranking.

### The Kaplan-Meier analysis is incomplete and its event definition is wrong

The KM cell did not run to completion, so the repository contains no survival
curves or log-rank statistic. Separately, its event definition uses a string
comparison that misclassifies `XX` (unknown) and `RA` (REO acquisition) as
defaults, and it queries the full performance panel without applying the 10% loan
sample. All three need fixing before the analysis is reported.

### The Monte Carlo simulation understates tail risk, substantially

Four specific issues:

1. **Volatility is assumed, not estimated.** The 25% figure is a hand-set parameter
   with no empirical basis. It should be estimated from the historical dispersion
   of realized annual loss rates, which the data supports.
2. **Buckets are drawn independently.** Credit losses are strongly correlated —
   recessions hit every FICO band at once. Independent draws let the buckets
   diversify each other almost completely, which is why the 99% VaR sits only 32%
   above expected loss. Real mortgage portfolios show far heavier tails. A
   single-factor model with an asset correlation, or a copula, would be more
   defensible.
3. **The normal distribution is the wrong shape.** Credit loss distributions are
   right-skewed and fat-tailed. A normal centered on expected loss cannot produce a
   crisis scenario. The `max(0, ...)` floor is a patch for the same underlying
   problem — a distribution that can generate negative losses is not the right one.
4. **PD and LGD are point estimates.** They are held fixed across all 10,000
   scenarios, so parameter uncertainty contributes nothing to the distribution.

The simulation demonstrates the VaR machinery correctly. Its output should not be
read as a risk estimate for a real portfolio.

### The test period's delinquency data may be unreliable

A diagnostic on the test set found 46 true severe-delinquency events against 15,175
suspicious transitions straight to `XX`/`RA` with no prior delinquency, across
764,958 loans. The censoring logic classifies those jumps as terminations rather
than defaults, which is defensible. But a 330:1 ratio suggests a data-quality issue
in the recent vintages rather than genuine loan behavior, and it is unresolved.

### Other constraints

- **`PROXY_LTV` is not an LTV.** It is the ratio of current balance to original
  balance, so it tracks amortization, not property value. `ELTV` is the real
  mark-to-market measure but is sparser.
- **Sampling is not seeded.** `USING SAMPLE 10 PERCENT` and `RANDOM() < 0.10` have
  no fixed seed, so re-running the pipeline produces a different sample. Metrics are
  not exactly reproducible across runs.
- **The test period includes COVID-19.** Forbearance under the CARES Act altered the
  relationship between delinquency and default in ways the pre-2019 training data
  does not contain. Only 0.17% of test loan-months carry a forbearance flag, which
  is low enough to question whether the field fully captures pandemic-era
  interventions.
- **Macro features are not forecast.** The model consumes contemporaneous GDP,
  unemployment, and HPI. Forward-looking use requires forecasting those inputs,
  which introduces error not reflected in any metric above.

## Reproducing this work

### Requirements

```
duckdb
pandas
pyarrow
numpy
lightgbm
scikit-learn
shap
optuna
lifelines
fredapi
matplotlib
seaborn
scipy
joblib
```

### Setup

The Freddie Mac dataset requires registration and agreement to its terms; it is not
redistributed here. Download the quarterly archives along with the origination and
performance header files from the Freddie Mac Single-Family Loan-Level Dataset
portal.

A FRED API key is required for the macroeconomic series (free from the St. Louis
Fed). Set it in the environment:

```bash
export FRED_API_KEY="your_key_here"
```

Paths in the notebook are absolute Windows paths under `D:\AI\` and need to be
changed to match your environment.

### Pipeline order

1. **Convert raw archives to Parquet.** Walks the download directory, unzips, applies
   headers, writes per-quarter Parquet to `master_orig_dataset/` and
   `master_perf_dataset/`.
2. **Build the modeling sample.** The DuckDB query producing
   `dtsa_sample_forward_looking.parquet`. Roughly 28 minutes on 28 threads with a
   75GB memory limit; adjust both to your hardware.
3. **Pull FRED series and merge.** Produces `train_balanced.parquet`,
   `val.parquet`, and `test.parquet` under `dtsa_ml_ready/`.
4. **Train the 12-month model.** Roughly 10 minutes.
5. **Run the Optuna study** (optional; 100 trials) and retrain with the best
   parameters.
6. **Build the Monte Carlo inputs.** The DuckDB query producing
   `monthly_pd.parquet`, then run the simulation.

### Hardware

Developed on a 28-thread workstation with 75GB of available memory. The DuckDB
steps stream from Parquet and will run with less, more slowly. The modeling steps
load full splits into pandas — the 41.5M-row test set is the binding constraint and
needs roughly 16GB.

