import numpy as np
import pandas as pd
from preprocessing import country_mapping, gender_mapping, scaling
import pickle
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r'Skybug-Customer-Churn-Prediction\data\Churn_Modelling.csv')
max_vals, min_vals, features, data = scaling(data, ['CreditScore', 'Age', 'Balance', 'EstimatedSalary'], MinMaxScaler())

column_names = data.columns.values

def scale_inputs(features, max_vals, min_vals, data):
    for i in range(len(features)):
        data[features[i]] = (data[features[i]] - min_vals[i]) / (max_vals[i] - min_vals[i])
    return data
#______________________________________________________________________________

# TAKING INPUTS

print("Enter the following details to predict the churn probability of the customer:")

country = input("Country (France, Spain, Germany): ")
while(country not in ["France", "Spain", "Germany"]):
    print("ERROR: Kindly enter valid country name.")
    country = input("Country (France, Spain, Germany): ")

credit_score = float(input("Credit Score: "))

gender = input("Gender (Male, Female) :")
while(gender  not in ["Male", "Female"]):
    print("ERROR: Kindly enter valid gender.")
    gender = input("Gender (Male, Female) :")

age = int(input("Age: "))
tenure = int(input("Tenure: "))
balance = float(input("Balance: "))
num_of_products = int(input("Number of Products: "))

has_credit_card = int(input("Has Credit Card (Yes:1, No:0): "))
while(has_credit_card  not in [1,0]):
    print("ERROR: Kindly enter valid credit card status.")
    has_credit_card = int(input("Has Credit Card (Yes:1, No:0): "))

is_active_member = int(input("Is Active Member (Yes:1, No:0): "))
while(is_active_member  not in [1,0]):
    print("ERROR: Kindly enter valid membership status.")
    is_active_member = int(input("Is Active Member (Yes:1, No:0): "))

estimated_salary = float(input("Estimated Salary: "))
print("\n")

#______________________________________________________________________________

# PREPROCESSING INPUTS

input_data = [credit_score, country, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary]

input_df= pd.DataFrame([input_data], columns = ['CreditScore', 'Geography', 'Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember', 'EstimatedSalary'])

scaled_df = scale_inputs(features, max_vals, min_vals, input_df)
scaled_df = country_mapping(scaled_df)
scaled_df = gender_mapping(scaled_df)

#______________________________________________________________________________

dtc = pickle.load(open(r'Skybug-Customer-Churn-Prediction\models\dtc.pkl', 'rb'))
lr = pickle.load(open(r'Skybug-Customer-Churn-Prediction\models\lr.pkl', 'rb'))
nb = pickle.load(open(r'Skybug-Customer-Churn-Prediction\models\nb.pkl', 'rb'))
svm_lin = pickle.load(open(r'Skybug-Customer-Churn-Prediction\models\SVM_lin.pkl', 'rb'))
svm_sig = pickle.load(open(r'Skybug-Customer-Churn-Prediction\models\SVM_sig.pkl', 'rb'))
xgb = pickle.load(open(r'Skybug-Customer-Churn-Prediction\models\xgb.pkl', 'rb'))

dtc_pred = dtc.predict(scaled_df)
lr_pred = lr.predict(scaled_df)
nb_pred = nb.predict(scaled_df)
svm_lin_pred = svm_lin.predict(scaled_df)
svm_sig_pred = svm_sig.predict(scaled_df)
xgb_pred = xgb.predict(scaled_df)

print("PREDICTIONS: (1 = Churn, 0 = No Churn)\n")
print("Decision Tree Classifier: ", dtc_pred[0])
print("Logistic Regression: ", lr_pred[0])
print("Naive Bayes: ", nb_pred[0])
print("Support Vector Machine (Linear Kernel): ", svm_lin_pred[0])
print("Support Vector Machine (Sigmoid Kernel): ", svm_sig_pred[0])
print("XGBoost: ", xgb_pred[0])