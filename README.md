# Network-Intrusion-Detection-

Abstract
With the rapid growth of internet usage, cybersecurity threats such as network intrusions and cyber attacks have increased significantly. This project presents a machine learning–based intrusion detection system that identifies malicious network traffic using classification algorithms. The system uses the CICIDS-2017 dataset developed by the Canadian Institute for Cybersecurity to detect attack patterns. Various machine learning models, including Isolation Forest, Random Forest, and Logistic Regression, are implemented and evaluated. The results demonstrate the effectiveness of machine learning techniques in detecting network intrusions. Data preprocessing, feature analysis, and performance visualization were also conducted to improve detection accuracy and interpretability.

1. Introduction
Cybersecurity has become a critical concern for organizations due to increasing cyber attacks such as denial-of-service attacks, data breaches, and network intrusions. Traditional rule-based intrusion detection systems often fail to detect new or unknown attacks.
Machine learning provides a data-driven approach to identify hidden patterns in network traffic and detect malicious activities automatically. This project aims to develop an intelligent intrusion detection system using machine learning algorithms to classify network traffic as normal or attack.
The main objectives of this project include:
•	Detect malicious network traffic using machine learning.
•	Compare performance of different classification models.
•	Extract important features contributing to attack detection.
•	Visualize model performance using Tableau Software dashboards.

2. Dataset Description
The project uses the CICIDS-2017 dataset developed by the Canadian Institute for Cybersecurity.
Dataset Characteristics
•	Contains network traffic flow records.
•	Includes both normal and attack traffic.
•	Provides multiple network features such as packet length, flow duration, and byte rate.
•	Designed for intrusion detection research.
Target Variable
•	BENIGN → 0 (Normal traffic)
•	Attack → 1 (Malicious traffic)

3. Methodology
The project follows a data mining and machine learning pipeline consisting of data preprocessing, model training, and performance evaluation.

3.1 Data Preprocessing
The dataset was cleaned and prepared using the following steps:
•	Removed missing values and infinite values.
•	Converted categorical labels into binary values.
•	Selected only numerical features.
•	Removed duplicate records to prevent data leakage.
•	Removed highly correlated features to reduce overfitting.
•	Scaled features using standard normalization.
Data preprocessing ensures reliable model training and improves performance.

3.2 Data Splitting
A time-based data splitting approach was used:
•	70% data → training set
•	30% data → testing set
This prevents information leakage and simulates real-world intrusion detection.

3.3 Machine Learning Models
Three models were implemented.
1. Isolation Forest
•	Unsupervised anomaly detection method.
•	Trained only on normal traffic.
•	Detects unusual network behavior.
2. Random Forest
•	Ensemble learning algorithm.
•	Uses multiple decision trees.
•	Handles class imbalance using class weights.
3. Logistic Regression
•	Supervised classification algorithm.
•	Predicts probability of network attack.

3.4 Feature Importance Analysis
Random Forest was used to identify the most important features responsible for detecting network attacks. This helps in understanding model decisions and improves interpretability.

3.5 Performance Evaluation Metrics
The models were evaluated using:
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	ROC-AUC score
•	Confusion matrix
These metrics measure the effectiveness of attack detection.


4. Data Visualization
Model performance and results were visualized using dashboards in Tableau Software.
The dashboard includes:
•	Model performance comparison
•	Attack vs normal traffic distribution
•	Feature importance visualization
•	Prediction accuracy analysis
Visualization helps interpret model performance and supports decision-making.

5. Applications
This system can be applied in:
•	Network security monitoring
•	Cyber attack detection
•	Enterprise security systems
•	Threat analysis platforms

6. Conclusion
This project demonstrates that machine learning techniques can effectively detect network intrusions and classify malicious traffic. The implemented models successfully identified attack patterns in network data. Data preprocessing and feature selection improved model performance, while visualization tools enhanced interpretability. Future improvements may include deep learning methods and real-time intrusion detection systems.

7. Future Work
•	Real-time intrusion detection system implementation.
•	Use of deep learning models.
•	Hyperparameter optimization.
•	Deployment in cloud environments.
•	Explainable AI techniques for attack interpretation.

8. References
1.	Canadian Institute for Cybersecurity — CICIDS-2017 Dataset
2.	Scikit-learn Machine Learning Library
3.	Tableau Software Documentation
