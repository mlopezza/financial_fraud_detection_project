# Financial Fraud Detection Project
- Detecting financial fraud using data analysis and machine learning techniques. Includes data preprocessing, feature engineering, model training, and evaluation to identify anomalous or high-risk financial transactions.

## Repository Structure
    ├── data
    ├──── processed
    ├──── raw
    ├── experiments
    ├── images
    ├── models
    ├── reports
    ├── src
    ├── README.md
    └── .gitignore

- Data: raw, processed and final data.
- Experiments: Experiments.
- Images: Final images.
- Models: Trained models or model predictions.
- Reports: Generated HTML, PDF etc. of the analysis report.
- src: Project source code.
- README: This file.
- .gitignore: Files to exclude from this folder.

## Team Members
- Lindsay Hudson
- Mariluz Lopez Zamora
- Joshua Okojie

## Chosen Dataset: Financial Transactions Dataset for Fraud Detection 
- URL: https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection/data

## Project Overview
- Purpose and Overview
- Methodology
- Data Analysis
- Predictive Model
- Technical Stack
- References



### Purpose and Overview
#### Business Problem
Financial fraud has increased substantially in recent years, costing institutions and consumers hundreds of millions of dollars annually (Hilal et al. 2022). In response, the financial sector has implemented increasingly sophisticated prevention measures, including fraud-detection systems that rely on anomaly-detection techniques to identify unusual or suspicious behavior. Over the past several decades, these methods have advanced significantly, driven by progress in statistical modeling, artificial intelligence, and machine learning (ML). Among the various types of financial fraud, credit card fraud remains one of the most prevalent and costly, making it a major priority for financial institutions.

This project focuses on developing a machine learning model capable of accurately detecting fraudulent credit card transactions, enabling faster identification, intervention, and protection for all stakeholders.



### Dataset Fraud Detection Scores: 
There are 3 different types of scores in the dataset: spending_deviation_score, velocity_score, geo_anomaly_score. 
- **Velocity Score:** 
        The transaction velocity, in a fraud context, is calculated by the number of transactions the took place in an account during a pre- specified timeframe (Wiese & Omlin, 2009). Different velocities can be calculated by grouping certain merchants into one velocity calculation. The score used to be calculated by the  Count transactions per unit of time and compare with historical averages, A high velocity score means an Unusual rapid activity as a potential card theft or automated fraud.  A Low score use to mean a normal transaction pace. 


            **** As previously discussed, there is also a cost for undetected fraud over time, which means that the value of fraud detection is a function of time (Bhattacharyya et al., 2011). The sooner the detection of fraudulent activity, the less the potential losses by individuals and companies. This is especially important to keep in mind as the typical fraudster has been known to exploit credit cards by spending as much as possible in as little time as possible until the fraud is detected and the card is deactivated (Bolton & Hand, 2002). 
- **Spending Deviation Score:**
    It measure How unusual a transaction amount is compared to the customer’s historical spending.  Its calculation is fften based on statistical deviation (e.g., z‑score).  A  High score means a transaction amount is far from normal as a possible fraud. A Low score means a transaction is consistent with past behavior.

- **Geo Anomaly Score:**



Tasks: 
- Domain Research & Problem Framing
- Study fraud detection techniques in financial services
- Identify business objectives and success metrics
- Research regulatory and compliance requirements
- Define scope of fraud types (transactional, identity theft, etc.)

# Questions: 
1. Types of transaction_type
2. Types of merchant_category
3. Types of locations
4. Types of device_used
5. Types of fraud_type
6. measure of time_since_last_transaction : sec, min, hours?
7. What is spending_deviation_score?
8. velocity_score measure?
9. What is geo_anomaly_score?
10. Types of payment_channel?



## Data information from kaggle
Transaction Details: ID, timestamp, sender/receiver accounts, amount, type (deposit, transfer, etc.)
Behavioral Features: time since last transaction, spending deviation score, velocity score, geo-anomaly score
Metadata: location, device used, payment channel, IP address, device hash
Fraud Indicators: binary fraud label (is_fraud) and type of fraud (e.g., money laundering, account takeover)
The dataset follows realistic fraud patterns and behavioral anomalies, making it suitable for:
Binary and multiclass classification models
Fraud detection systems
Time-series anomaly detection
Feature engineering and model explainability




---

## 3. Geo Anomaly Score
- **Measures:** It is a measure of Geographic inconsistencies in transaction locations. It use to Compare current transaction location with previous ones; check distance and time feasibility.  For example purchase in Toronto followed by another in Tokyo within 10 minutes.  A  High score means an impossible or improbable travel as a suspicious activity.  
  - Low score → location consistent with customer’s usual pattern.

---

##  Summary
- **Spending Deviation Score** → detects abnormal amounts.  
- **Velocity Score** → detects abnormal transaction frequency.  
- **Geo Anomaly Score** → detects abnormal geographic patterns.  
Together, they form part of a broader **fraud risk model** that flags transactions for review or rejection.


##### Features description

### Methodology

### Data Analysis
### Predictive Model
#### Model Purpose
#### Building the Model
#### Model Performance
#### Model Conclusion and Next Steps
#### Project Scope
##### Stakeholders

### Technical Stack
#### Programming Language
- Python
- SQL

#### Libraries Used
- NumPy: matrix operations, numerical computations
- Pandas: data analysis, handling datasets
- SciPy: statistical tests and scientific computations
- Statsmodels: statistical modeling and regression analysis
- Matplotlib: creating graphs and plots -Seaborn: enhancing matplotlib plots, data visualization
- Plotly: interactive graphs and plots
- scikit-learn (SKLearn): machine learning, preprocessing, classification, pipelines, model evaluation
- PIL (Pillow): image processing
- Requests: HTTP requests, downloading data from the web

### References
- 1. Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances
Hilal et al. - Expert Systems with Applications - 2022, https://doi.org/10.1016/j.eswa.2021.116429
