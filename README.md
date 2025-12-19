<<<<<<< HEAD
# Financial Fraud Detection Project
Detecting financial fraud using data analysis and machine learning techniques. Includes data preprocessing, feature engineering, model training, and evaluation to identify anomalous or high-risk financial transactions.

## Type of project
- Data exploration with SQL
- Data visualization wiht Python
- Predictive model

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

- **Spending Deviation Score:**
    It measure how unusual a transaction amount is compared to the customer’s historical spending.  Its calculation is often based on statistical deviation (e.g., z‑score).  A  High score means a transaction amount is far from normal as a possible fraud. A Low score means a transaction is consistent with past behavior.

- **Geo Anomaly Score:**
    It is a measure of Geographic inconsistencies in transaction locations. It use to Compare current transaction location with previous ones; check distance and time feasibility.  For example purchase in Toronto followed by another in Tokyo within 10 minutes.  A  High score means an impossible or improbable travel as a suspicious activity.  A low score means that location consistent with customer’s usual pattern.






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



##### Features description

## Data Dictionary

## Data Dictionary

| Feature | Type | Description |
| :--- | :--- | :--- |
| `transaction_id` | VARCHAR | Unique identifier for each transaction. |
| `timestamp` | TIMESTAMP | Date and time the transaction occurred. ISO8601 |
| `sender_account` | VARCHAR | Sender account number. |
| `receiver_account` | VARCHAR | Destination account number. |
| `amount` | DOUBLE | Transaction value. |
| `transaction_type` | VARCHAR | Transfer, Withdrawal, Payment, Deposit. |
| `merchant_category` | VARCHAR | Restaurant, travel, other, retail, online, entertainment, utilities, grocery. |
| `location` | VARCHAR | London, Sydney, New York, Berlin, Tokyo, Dubai, Singapore, Toronto. |
| `device_used` | VARCHAR | mobile, atm, web, pos. |
| `is_fraud` | BOOLEAN | Binary flag indicating fraud (1) or legitimate (0). |
| `fraud_type` | VARCHAR | Card not present, Nome. |
| `time_since_last_transaction` | DOUBLE | Time elapsed since the user's previous transaction. |
| `spending_deviation_score` | DOUBLE | Score representing deviation from typical spending habits. |
| `velocity_score` | BIGINT | Measure of transaction frequency in a short window. |
| `geo_anomaly_score` | DOUBLE | Score based on unusual distance between transactions. |
| `payment_channel` | VARCHAR | wire_transfer, ACH, card, UPI. |
| `ip_address` | VARCHAR | IP address from which the transaction was initiated. |
| `device_hash` | VARCHAR | Unique digital fingerprint of the hardware. |



### Methodology
#### Exploratory Data Analysis via SQL
1. The data set spans the period of one year from 2023-01-01 to 2024-01-01
1. There weren0t found any ramdom spaces
2.  Missing and Null Values were found in 2 columns: 
    - fraud_type_nulls = 4820447
    - time_since_last_transaction_nulls = 896513
    - However, there where not found any null value between the fraud_cases positive. 
3. The numner of positive fraud cases were: 179553 with negative fraud cases = 4820447
4. From the all positive fraud cases there were divided as follow: 
    - wire_transfer	45034
    - UPI	44896
    - card	44885
    - ACH	44738
5. From Fraud Cases positive: 
   - Min amount = 0.01 and Max Amount = 	3128.14
   - On time since last transaction were found some negative values, we didn't found any information on the data set about how it was calculated or why there are some negative values, a posible conclusion is that this negative values means the time difference between diferent locations?
        - Min time since last transaction: -8748.166439	(89880 fraud cases with negative value) and maximum time since last transaction = 8744.774704 (89673 fraud cases with positive value)

6. There were found repetition in fraud cases positive when we look for sender acound and receiver acount
    - Sender Acount with fraud positive: Min repetitions: 2,  max repetitions: 7
    - Receiver Acount with fraud positive:  Min repetitions: 2 , max repetitions: 5


##### Assesment of distinct values by column:
- transaction_id: 5 million unique values, this is logical as there are 5 million rows and each transaction should have its own unique value.

- timestamp: 4,999,998 unique values, no nulls values so that means there is two transactions that occurred at the same time as other transactions. To be broken down into month, day of week, hour during feature engineering.

- sender_account: 896,513 unique values (may be hashed for PI reasons)

- receiver_account: 896639 unique values (may be hashed for PI reaons)

- amount: 217,068 unique values that range from 0.01 to 3520.57. We may want to consider converting amount into ranges or categories of some sort when feature engineering.

- transaction_type: 4 unique values; deposit, payment, transfer, withdrawal

- merchant_category: 8 unique values; entertainment, grocery, online, other, restaurant, retail, travel, utilities

- location: 8 unique values; Berlin, Dubai, London, New York, Singapore, Sydney, Tokyo, Toronto

- device_used: 4 unique values; atm, mobile, pos, web

- is-fraud: 2 unique values; 0 = false and 1 = true

- fraud_type: 2 unique values; card_not_present and none. This column offers little value - to be deleted.

- time_since_last transaction: 4,103,488 unique values. Ranges from -8777.814182 to 8757.758483 We may want to convert into range or categories of some sort when feature engineering (ex. less than one minute, less than 5 minutes etc).

- spending_deviation_score: 917 unique values; raning from -5.26 to 5.02

- velocity_score: 20 unique values; ranges from 1-20

- geo_anomaly_score: 101 unique values; ranges from 0-1 (decimal values)

- payment_channel: 4 unique values; ACH, UPI, card, wire_transfer

- ip_address: 4,997,068 unique values (to be hashed for PI reasons)

- device_hash: 3,835,723 unique values



### Data Analysis

 **** As previously discussed, there is also a cost for undetected fraud over time, which means that the value of fraud detection is a function of time (Bhattacharyya et al., 2011). The sooner the detection of fraudulent activity, the less the potential losses by individuals and companies. This is especially important to keep in mind as the typical fraudster has been known to exploit credit cards by spending as much as possible in as little time as possible until the fraud is detected and the card is deactivated (Bolton & Hand, 2002). 

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
=======
# financial_fraud_detection_project
Detecting financial fraud using data analysis and machine learning techniques.
>>>>>>> 4f8ef76 (Initial commit)
