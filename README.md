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
The dataset selected from Kaggle consists of 5 million synthetically generated financial transactions. It is designed to simulate real-world transactional behavior for fraud detection research and machine learning applications.

The dataset includes 18 attributes, among them the target variable is_fraud and three types of anomaly scores: spending_deviation_score, velocity_score, and geo_anomaly_score.

- **Velocity Score:** 
    The score is typically calculated by counting the number of transactions per unit of time and comparing it with historical averages. 
    
    A high velocity score indicates unusually rapid activity, which may suggest potential card theft or automated fraud. A low score generally reflects a normal transaction pace.

- **Spending Deviation Score:**
    It measures how unusual a transaction amount is compared to the customer’s historical spending. 
    
    A high score indicates that the transaction amount is far from the customer’s normal pattern and may signal potential fraud. A low score suggests the transaction is consistent with past behavior.

- **Geo Anomaly Score:**
    It is a measure of geographic inconsistencies in transaction locations. It compares the current transaction location with previous ones and checks whether the distance and timing are feasible. For example, a purchase in Toronto followed by another in Tokyo within 10 minutes would be flagged.
    
    A high score indicates impossible or highly improbable travel and is therefore suspicious. A low score means the transaction location is consistent with the customer’s usual pattern.
   


### Feature description: 

| Feature                       | Type      | Distinct Values | Description                                              | Notes                                                                 |
|-------------------------------|-----------|-----------------|----------------------------------------------------------|-----------------------------------------------------------------------|
| transaction_id                | VARCHAR   | 5,000,000       | Unique identifier for each transaction.                  | All unique.                                                           |
| timestamp                     | TIMESTAMP | 4,999,998       | Date and time the transaction occurred (ISO8601).        | Two timestamps are duplicated; no nulls. Useful for extracting month/day/hour.  |
| sender_account                | VARCHAR   | 896,513         | Sender account number (hashed).                           | High cardinality.                                                     |
| receiver_account              | VARCHAR   | 896,639         | Destination account number (hashed).                     | High cardinality.                                                     |
| amount                        | DOUBLE    | 217,068         | Monetary value of the transaction.                       | Ranges from 0.01 to 3520.57; may be bucketed into ranges. |
| transaction_type              | VARCHAR   | 4               | deposit, payment, transfer, withdrawal.                  | Categorical.                                                          |
| merchant_category             | VARCHAR   | 8               | entertainment, grocery, online, other, restaurant, retail, travel, utilities. | Categorical.                                      |
| location                      | VARCHAR   | 8               | Berlin, Dubai, London, New York, Singapore, Sydney, Tokyo, Toronto. | Geographic categorical.                           |
| device_used                   | VARCHAR   | 4               | atm, mobile, pos, web.                                   | Device used to initiate the transaction.                              |
| is_fraud                      | BOOLEAN   | 2               | Binary flag indicating fraud (1) or legitimate (0).      | Binary target variable.                                               |
| fraud_type                    | VARCHAR   | 2               | card_not_present, none.                                  | Very low value; candidate for removal or merging.                     |
| time_since_last_transaction   | DOUBLE    | 4,103,488       | Time elapsed since the user's previous transaction.      | Ranges from -8777.81 to 8757.76; may be bucketed into time ranges or outlier handling. |
| spending_deviation_score      | DOUBLE    | 917             | Deviation from typical spending habits.                  | Ranges from -5.26 to 5.02. Continuous.        |
| velocity_score                | BIGINT    | 20              | Measure of transaction frequency in a short window.      | Discrete range from 1 to 20.                                                  |
| geo_anomaly_score             | DOUBLE    | 101             | Score based on unusual distance between transactions.    | Ranges from 0 to 1 (decimal).                         |
| payment_channel               | VARCHAR   | 4               | ACH, UPI, card, wire_transfer.                           | Categorical.                                                          |
| ip_address                    | VARCHAR   | 4,997,068       | IP address from which the transaction was initiated (hashed).  | Very high cardinality.                                                |
| device_hash                   | VARCHAR   | 3,835,723       | Unique digital fingerprint of the hardware (hashed).     | Very high cardinality.                                                |



### Methodology
#### Data Exploration
The downloaded CSV file containing the original dataset was converted into columnar Parquet files, which are much faster to query. After that, data exploration and cleaning were performed using SQL queries in DuckDB to improve memory efficiency.

The initial data exploration revealed that the dataset has a significant class imbalance. Out of 5 million transactions, the number of positive fraud cases is 179,553, while negative (non-fraud) cases total 4,820,447, resulting in a fraud ratio of 0.035911 and a non-fraud ratio of 0.964089.

The dataset spans a one-year period, from 2023-01-01 to 2024-01-01. In addition, all columns are stored in a consistent internal format, and no random spaces were found; therefore, no adjustments were required.


**Missing and Null Values**

Data exploration identified two features with NULL values: time_since_last_transaction (896,513) and fraud_type (4,820,447). 

However, no NULL values were found among positive fraud cases. All NULL values belong to the is_fraud = FALSE category, as this group contains the largest number of observations, with a non-fraud transaction ratio of 0.96 compared to a fraud transaction ratio of 0.035. Therefore, removing records with NULL values does not affect the minority class, which is also the class of interest for identifying fraud patterns.


**Identifier Features for Future Anonymization**

sender_account, receiver_account, transaction_id, ip_address and device_hash.


**Unique Values and repetitions:**

Repeated values were found among the identifier features. Sender_account and receiver_account showed potential anomalies, with 896,513 and 896,639 unique values, respectively, in a dataset of 5 million transactions. For this reason, a deeper analysis was performed focusing on fraud-positive cases.

In the fraud-positive transactions, sender_account had 16,337 repeated values, with a maximum of 7 repetitions, while receiver_account had 15,604 repeated values, with a maximum of 5 repetitions. 

These results suggest that fraudulent activity is highly concentrated in specific accounts.

Both sender_account and receiver_account show a highly skewed distribution. Most accounts appear only once or twice, while very few accounts appear multiple times. For example, in sender_account, only one value appears 7 times and two values appear 5 times, compared to more than 145,000 values that appear only once. A similar pattern is observed for receiver_account, indicating that repeated accounts are extremely rare and that the dataset is dominated by unique or low-frequency account identifiers.

**Negative values on time_since_last_transaction**

Negative values were found in the time_since_last_transaction feature. The dataset does not provide information about how this variable was calculated or why negative values exist.

The minimum value is -8748.17 (with 89,880 fraud cases showing negative values), and the maximum value is 8744.77 (with 89,673 fraud cases showing positive values). These values are close to the approximate number of hours in a year (8,760), but negative values are illogical because they would imply that some transactions occurred in the future relative to previous ones.  

Additional analyses were performed to understand this behavior. First, it was tested whether negative values were related to specific geographic locations, possibly due to time zone differences, but no correlation was found.

Finally, transactions were grouped by sender_account, ordered by timestamp, and the time differences were recalculated. This approach also showed no meaningful pattern, likely because the dataset does not contain complete transaction histories for each user. As a result, this feature cannot be reliably reconstructed or interpreted.

**Other Features**

Fraud cases were found across all payment_channel categories, which indicates that all categories are significant.
From Fraud Cases positive: the Min amount was 0.01 and Max Amount was	3128.14

### Data cleaning
Data cleaning was conducted with SQL queries and The cleaned table was saved as a Parquet file for modeling.

#### Feature engineering: 
-  timestamp was divided in diferent columns: month, day, hour.
- it was creates a new column for Day of the week using ISODOW format. 

#### Drop features: 
-  timestamp. 
-  fraud_type
-  transaction_id
-  NULL rows from time_since_last_transaction, after that there were 3923934 negative fraud cases and positive fraud cases initial number: 179553 were not altered. 

The ip_address and device_hash features were removed considering:
- Both are cardinal columns
- ip_adress had only	6 repetitie values in fraud cases and a maximum of 2 repetitions.
- device_hash	had 1,757 repeated values and a maximum of 3 repetitions.
- Fraudulent activity is moderately concentrated in certain devices and minimally traceable through IP addresses. Given the dataset size, both features contribute little information, and their removal reduces noise and dimensionality.

#### Final Features Selected: 
- sender_account
- receiver_account
- amount
- transaction_type
- merchant_category
- location
- device_used
- is_fraud
- time_since_last_transaction
- spending_deviation_score
- velocity_score
- geo_anomaly_score
- payment_channel
- year
- month
- day_of_month
- hour
- day_of_week

### Data Analysis

 **** As previously discussed, there is also a cost for undetected fraud over time, which means that the value of fraud detection is a function of time (Bhattacharyya et al., 2011). The sooner the detection of fraudulent activity, the less the potential losses by individuals and companies. This is especially important to keep in mind as the typical fraudster has been known to exploit credit cards by spending as much as possible in as little time as possible until the fraud is detected and the card is deactivated (Bolton & Hand, 2002). 

VEL score:  The transaction velocity, in a fraud context, is calculated by counting the number of transactions that take place in an account during a pre‑specified timeframe (Wiese & Omlin, 2009). Different velocity measures can be created by grouping certain merchants into a single velocity calculation.


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

