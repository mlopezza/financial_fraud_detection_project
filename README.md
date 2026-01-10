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
- Mariluz Lopez Zamora
- Joshua Okojie
- Lindsay Hudson


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
The dataset selected from Kaggle contains 5 million synthetically generated financial transactions. It is designed to simulate real‑world behavior for fraud detection research and machine learning applications.

The dataset includes 18 attibutes, among them the target variable is_fraud and three different types of anomaly scores: spending_deviation_score, velocity_score, and geo_anomaly_score.

- **Velocity Score:** 
    The transaction velocity, in a fraud context, is calculated by counting the number of transactions that take place in an account during a pre‑specified timeframe (Wiese & Omlin, 2009). Different velocity measures can be created by grouping certain merchants into a single velocity calculation.

    The score is typically calculated by counting the number of transactions per unit of time and comparing it with historical averages. A high velocity score indicates unusually rapid activity, which may suggest potential card theft or automated fraud. A low score generally reflects a normal transaction pace.

- **Spending Deviation Score:**
    It measures how unusual a transaction amount is compared to the customer’s historical spending. Its calculation is often based on statistical deviation (e.g., a z‑score). 
    
    A high score indicates that the transaction amount is far from the customer’s normal pattern and may signal potential fraud. A low score suggests the transaction is consistent with past behavior.

- **Geo Anomaly Score:**
    It is a measure of geographic inconsistencies in transaction locations. It compares the current transaction location with previous ones and checks whether the distance and timing are feasible. For example, a purchase in Toronto followed by another in Tokyo within 10 minutes would be flagged.
    
    A high score indicates impossible or highly improbable travel and is therefore suspicious. A low score means the transaction location is consistent with the customer’s usual pattern.
   


### Features description

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
#### Data Exploration
The downloaded CSV file containing the original dataset was converted into columnar Parquet files, which are much faster to query. After that, data exploration and cleaning were performed using SQL queries in DuckDB to improve memory efficiency.

First data exploration found that the data set spans the period of one year from 2023-01-01 to 2024-01-01. In adition, the timestamp column is stored in a consistent internal format (as TIMESTAMP type) therefore it does not need to be adjusted.


- All columns are stored in a consistent internal format, therefore it does not need to be adjusted.
- Unique Values: 
Distinct Values by Column

| Column                        | Distinct Values | Notes / Description |
|-------------------------------|-----------------|---------------------|
| transaction_id                | 5,000,000       | Each transaction has a unique ID. |
| timestamp                     | 4,999,998       | Two timestamps are duplicated; no nulls. Useful for extracting month/day/hour. |
| sender_account                | 896,513         | Likely hashed for privacy. |
| receiver_account              | 896,639         | Likely hashed for privacy. |
| amount                        | 217,068         | Ranges from 0.01 to 3520.57; may be bucketed into ranges. |
| transaction_type              | 4               | deposit, payment, transfer, withdrawal. |
| merchant_category             | 8               | entertainment, grocery, online, other, restaurant, retail, travel, utilities. |
| location                      | 8               | Berlin, Dubai, London, New York, Singapore, Sydney, Tokyo, Toronto. |
| device_used                   | 4               | atm, mobile, pos, web. |
| is_fraud                      | 2               | 0 = non‑fraud, 1 = fraud. |
| fraud_type                    | 2               | card_not_present, none. Low value; candidate for removal. |
| time_since_last_transaction   | 4,103,488       | Ranges from -8777.81 to 8757.76; may be bucketed into time ranges. |
| spending_deviation_score      | 917             | Ranges from -5.26 to 5.02. |
| velocity_score                | 20              | Ranges from 1 to 20. |
| geo_anomaly_score             | 101             | Ranges from 0 to 1 (decimal). |
| payment_channel               | 4               | ACH, UPI, card, wire_transfer. |
| ip_address                    | 4,997,068       | Likely hashed for privacy. |
| device_hash                   | 3,835,723       | Likely hashed for privacy. |


#### Check the relevance of repeat values on sender_account, receiver_account, ip_address, device_hash
6. There were found repetition in fraud cases positive when we look for sender acound and receiver acount
    - Sender Acount with fraud positive: Min repetitions: 2,  max repetitions: 7
    - Receiver Acount with fraud positive:  Min repetitions: 2 , max repetitions: 5
sender_account (896,513 unique values) and receiver_account (896,639 unique values) come from a dataset of 5 million transactions. It is important to check whether these values repeat within the fraud‑positive cases. The same applies to ip_address (4,997,068 unique values) and device_hash (3,835,723 unique values).

There were repeated values found in several features: sender_account had 16,337 repeated values with a maximum of 7 repetitions; receiver_account had 15,604 repeated values with a maximum of 5 repetitions; ip_address had 6 repeated values with a maximum of 2 repetitions; and device_hash had 1,757 repeated values with a maximum of 3 repetitions.

The results suggest that fraudulent activity in this dataset is highly concentrated in specific accounts, moderately concentrated in certain devices, and minimally traceable through IP addresses, which aligns with typical fraud behaviors involving account compromise, device reuse, and IP obfuscation.

Both sender_account and receiver_account show a highly skewed distribution of repetitions. The vast majority of accounts appear only once or twice, while very few accounts repeat multiple times. In sender_account, only one value appears 7 times and two values appear 5 times, compared with more than 145,000 values that appear only once. A similar pattern is observed in receiver_account, where just one value appears 5 times and most values occur only once. This indicates that repeated accounts are extremely rare, and the dataset is dominated by unique or low‑frequency account identifiers.

There were not found any ramdom spaces. 

2.  Missing and Null Values were found in 2 columns: 
    - fraud_type_nulls = 4820447
    - time_since_last_transaction_nulls = 896513
    - However, there where not found any null value between the fraud_cases positive. 
    Data exploration identified two (2) features with NULL values: time_since_last_transaction (896513) and fraud_type (4820447). Queries were conducted to understand the number of NULL values ​​per feature and the proportion of NULL and non-NULL values ​​per feature to determine the best way to address the NULL values.

    All NULL values are in the "is_fraud = FALSE" category because this group contains the largest number of observations, with a non‑fraud transaction ratio of 0.96 versus a fraud transaction ratio of 0.035. Therefore, dropping the NULL values does not affect the minority class which is also the clase of interest to found patterns.


3. The numner of positive fraud cases were: 179553 with negative fraud cases = 4820447, with fraud_ratio = 0.035911 and no_fraud_ratio = 0.964089
4. From the all positive fraud cases there were divided as follow payment_channel: 
    - wire_transfer	45034
    - UPI	44896
    - card	44885
    - ACH	44738
    Fraud cases were found across all payment_channel categories, which indicates that all categories are significant.

5. From Fraud Cases positive: 
   - Min amount = 0.01 and Max Amount = 	3128.14
   - On time since last transaction were found some negative values, we didn't found any information on the data set about how it was calculated or why there are some negative values, a posible conclusion is that this negative values means the time difference between diferent locations?
        - Min time since last transaction: -8748.166439	(89880 fraud cases with negative value) and maximum time since last transaction = 8744.774704 (89673 fraud cases with positive value)

Negative values were found only in the time_since_last_transaction feature, ranging from -8777.814182 to 8757.758483. According to the dataset authors, this variable represents the number of hours since the user’s previous transaction. The values may reference the approximate number of hours in a year (8760 hours). However, the negative values are illogical—for example, they would imply that some transactions occur in the future relative to the previous one.

Additional queries were conducted to understand why these negative values appear. One possible explanation was that time differences might have been calculated using transactions recorded in different time zones. To test this, the analysis examined whether negative values were concentrated in specific geographic locations. No correlation was found between negative time_since_last_transaction values and any particular location.

Furthermore, in another attempt to interpret these negative values, the data was grouped by sender_account, ordered by timestamp, and the time difference between each transaction and the previous one was recalculated. This analysis also revealed no meaningful pattern, likely because the dataset does not include all transaction records for each user. As a result, there is no reliable timeline available to compute this feature accurately.






### Data Analysis

 **** As previously discussed, there is also a cost for undetected fraud over time, which means that the value of fraud detection is a function of time (Bhattacharyya et al., 2011). The sooner the detection of fraudulent activity, the less the potential losses by individuals and companies. This is especially important to keep in mind as the typical fraudster has been known to exploit credit cards by spending as much as possible in as little time as possible until the fraud is detected and the card is deactivated (Bolton & Hand, 2002). 


After data exploration with SQL and a exploratory data visualization, with literature review



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

# financial_fraud_detection_project
Detecting financial fraud using data analysis and machine learning techniques.

