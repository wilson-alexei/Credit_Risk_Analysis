# Credit_Risk_Analysis
Utilize `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling to solve credit card risk

## Overview 
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, you’ll compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

> We need to create the training variables by converting the string values into numerical ones using the `get_dummies()` pandas method

<img width="1256" alt="Dummy_Data" src="https://user-images.githubusercontent.com/95068439/164836946-b4c8109f-d7eb-4403-aabe-2f44cab60382.png">

1. `RandomOverSampler`:

  -  Balanced Accuracy Score: 64.6 %
  -  Precision Score: 1 % [high_risk]
  -  Recall Score: 61 % [high_risk]

<img width="1103" alt="RANDOM_Acc_Score" src="https://user-images.githubusercontent.com/95068439/164837059-9bc481cf-44b9-437a-9fd8-6dcb4147d322.png">

<img width="1100" alt="RANDOM_CM_Report" src="https://user-images.githubusercontent.com/95068439/164837373-ce1f55ff-2023-4008-a019-df75ad33a89f.png">

2. `SMOTE`:

  - Balanced Accuracy Score: 62.3 %
  - Precision Score: 1 % [high_risk]
  - Recall Score: 61 % [high_risk]


<img width="1107" alt="SMOTE_Report" src="https://user-images.githubusercontent.com/95068439/164838035-9a44071c-b0c0-40b1-9ebb-022c6e761017.png">

3. `ClusterCentroids`:

  - Balanced Accuracy Score: 50.8 %
  - Precision Score: 1 % [high_risk]
  - Recall Score: 57 % [high_risk]

<img width="1086" alt="CENTROID_Report" src="https://user-images.githubusercontent.com/95068439/164838472-f12ca914-8b08-4069-8adf-01555a666182.png">

4. `SMOTEEN`:

  - Balanced Accuracy Score: 61.6 %
  - Precision Score: 1 % [high_risk]
  - Recall Score: 69 % [high_risk]

<img width="1083" alt="SMOTEEN_Report" src="https://user-images.githubusercontent.com/95068439/164838598-b1932874-1d11-4683-b002-b3d861f2f3bb.png">

5. `BalancedRandomForestClasifier`:

  - Balanced Accuracy Score: 78.8 %
  - Precision Score: 4 % [high_risk]
  - Recall Score: 67 % [high_risk]

<img width="1085" alt="BCRF_Report" src="https://user-images.githubusercontent.com/95068439/164840732-e9f3a6dd-2ee7-4858-b6bc-b1c09882f51b.png">

6. `EasyEnsembleClassifier`:

  - Balanced Accuracy Score: 92.5 %
  - Precision Score: 7 % [high_risk]
  - Recall Score: 91 % [high_risk]

<img width="1092" alt="EE_Report" src="https://user-images.githubusercontent.com/95068439/164841962-06ba2999-28b9-4a1e-97d1-8265025db45d.png">

## Summary 
Of all the 6 models, `EasyEnsembleClasifier` is the best with the highest accuracy, precision, recall, and F-1. However, I would not recommend Jill to use any of these models as the precision scores for the high risk applications are very low. This may also lead to high number of False Negatives which causes the bank to lose lots of good(low risk) potential applications which can cost them their potential stream of revenues. 

> LinkedIn: https://www.linkedin.com/in/wilson-alexei/

> Email: wils.alexei@gmail.com
