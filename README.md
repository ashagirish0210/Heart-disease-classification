# Heart-disease-classification
Heart disease prediction using 6 machine learning models with interactive Streamlit deployment

a) Problem statement

The objective of this project is to build multiple machine learning classification models to predict the presence of heart disease from clinical attributes, evaluate them using standard classification metrics, and deploy an interactive Streamlit web application for model demonstration and testing. 

b) Dataset description

Dataset: Heart Disease UCI dataset (CSV: `heart_disease_uci.csv`). 
Task type: Binary classification (Heart Disease: Yes/No). 
Target column: `num` (converted to binary where 0 = No disease and 1 = Disease when `num > 0`). 
Dataset has 16 features and 920 instances. 
Features:
id	age	sex	dataset	cp	trestbps	chol	fbs	restecg	thalch	exang	oldpeak	slope	ca	thal	num

Preprocessing 
Removed non‑predictive identifiers (`id`) and dataset source column (`dataset`) for training. 
Encoded categorical columns (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`) for model training.
Handled missing values using numeric median strategy.
Used scaling where required 

c) Models used (with evaluation metrics)

Models implemented :Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost. 

Metrics reported for each model: Accuracy, AUC Score, Precision, Recall, F1 Score, MCC. 

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8174	0.9034	0.8176	0.8174	0.8165	0.6296
Decision Tree	0.8261	0.8669	0.826	0.8261	0.8255	0.6473
kNN	0.8217	0.8889	0.8224	0.8217	0.822	0.6408
Naive Bayes	0.8261	0.8858	0.8259	0.8261	0.8259	0.6478
Random Forest (Ensemble)	0.8391	0.9217	0.8394	0.8391	0.8385	0.6739
XGBoost (Ensemble)	0.8478	0.9115	0.8487	0.8478	0.847	0.6919
Observations on model performance 

ML Model Name 	Observation about model performance 
Logistic Regression: Good baseline with balanced performance; AUC is strong (0.9034) showing good class separation, but overall accuracy/MCC are lower than ensemble models
Decision Tree:	Performs competitively with decent accuracy (0.8261) but the lowest AUC (0.8669), indicating weaker ranking/separation capability compared to other models
kNN: 	Similar to Logistic Regression; performs reasonably after scaling, with improved AUC (0.8889) but slightly lower MCC than Naive Bayes and ensembles
Naive Bayes:	Slight improvement over Decision Tree in AUC (0.8858) and best MCC among non‑ensemble models (0.6478), indicating stable performance despite simplifying assumptions
Random Forest (Ensemble):	Stronger overall than single models; achieves high AUC (0.9217) and higher MCC (0.6739), suggesting better generalization through bagging
XGBoost (Ensemble):Best overall results (highest Accuracy 0.8478 and highest MCC 0.6919); boosting captures feature interactions and non‑linear patterns effectively


