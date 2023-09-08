## Import Libraries ##
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, precision_recall_curve,auc, accuracy_score
import os
import time

## Data Understanding ##

# Data Loading
df = pd.read_csv('./cardio_train.csv', sep=";")
df.head()

# Look the info of the dataset
df.info()

# Look data description
df.describe().T

## EDA ##

# Checking missing value
df.isnull().sum()

# Checking the duplicated data
df.duplicated().sum()

# Drop featrue id
df.drop(['id'], axis=1, inplace=True)

# Create histogram of the numerical columns to know how data distributed for each feature

cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
plt.figure(figsize=(15,15))
for i, col in enumerate(cols):
    plt.subplot(3,4,i+1)
    plt.xlabel(f'{skew(df[col])}')
    plt.ylabel("Count", fontsize=13)
    plt.subplots_adjust(hspace=1, wspace=0.3)
    plt.grid(True)
    plt.title(col)
    sns.histplot(data=df, x=col, bins=50, kde=True)

# Check box plot for the numerical columns to find out whether there are outliers in each feature
cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
plt.figure(figsize=(20,20))
for i, col in enumerate(cols):
    plt.subplot(3,4,i+1)
    plt.title(col)
    plt.xlabel('Frequency', fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.subplots_adjust(hspace=1, wspace=0.3)
    sns.boxplot(data=df, x=col, palette='flare', fliersize=1)
    
# Remove outliers using IQR Technique
col_num = ['age','height', 'weight', 'ap_hi', 'ap_lo']
def RemoveOutliers(col_num):
    for col in col_num:
        iqr = df[col].quantile(0.75)-df[col].quantile(0.25)
        lower_threshold = df[col].quantile(0.25)-(iqr*1.5)
        upper_threshold = df[col].quantile(0.75)+(iqr*1.5)
        print(f"The range of outlier values in the feature {col} : <{round(lower_threshold,2)} or >{round(upper_threshold,2)}")
        df.loc[(df[col]<=lower_threshold),col] = lower_threshold
        df.loc[(df[col]>=upper_threshold),col] = upper_threshold
RemoveOutliers(col_num)

# Check box plot for the numerical columns to find out whether there are outliers in each feature
cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
plt.figure(figsize=(20,20))
for i, col in enumerate(cols):
    plt.subplot(4,3,i+1)
    plt.title(col)
    plt.xlabel('Frequency', fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.subplots_adjust(hspace=1, wspace=0.3)
    sns.boxplot(data=df, x=col, palette='flare', fliersize=1)
    
# Checking value counts each feature
cols = df.columns
for col in cols:
    print(f"{df[col].value_counts()}\n")
    

## Feature Age ##

# Umur yang dihitung berdasarkan hari akan di ubah menjadi tahun.
df['age'] = df['age'].apply(lambda x: x // 365)
df['age'] = df['age'].astype(int)

# Plot age
plt.figure(figsize=(12,8))
plt.xlabel('Frequency')
plt.ylabel("Count", fontsize=13)
plt.subplots_adjust(hspace=1, wspace=0.3)
plt.xticks(rotation=45)
plt.title("Age value counts")
plt.grid(True)
sns.barplot(x=df['age'].value_counts().index, y=df['age'].value_counts().values)

# Plot age terhadap cardio
dfage_agg = df.groupby('age')['cardio'].value_counts().unstack(fill_value=0)
list_noncardio = dfage_agg[0].tolist()
list_cardio = dfage_agg[1].tolist()
x = dfage_agg.index
# Set width to give a space for bar chart
width = 0.25
plt.figure(figsize=(12,8))
bar_noncar = plt.bar(x, list_noncardio, width, color='#468B97')
bar_car = plt.bar(x + width, list_cardio, width, color='#CD1818')
# Set the title, label, and legend
plt.xlabel('Age')
plt.ylabel("Count cardio vs non cardio")
plt.title("Cardio VS Non Cardio  (Aggregated by Age)")
plt.legend( (bar_noncar, bar_car ), ('Non Cardio', 'Cardio'))
# Adjustment plot
plt.grid(True, color = "grey", alpha=0.5, linestyle = "-")
plt.xticks(x+width/2,dfage_agg.index, rotation=90)
plt.show()

## Feature Gender ##

# Plot Gender terhadap cardio
labels = ["Male", "Female"]
value = df['gender'][df['cardio']==1].value_counts().tolist()
tot = sum(value)
percentages = [(value / tot) * 100 for value in value]
# Create a pie chart with value percentages inside
plt.pie(value, labels=labels, autopct='%.1f%%', startangle=140)
# Draw center circle to make it look like a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title("Male VS Female suffer from cardiovascular disease")
plt.show()

## Feature Cholesterol & Gluc

# Plot chol terhadap cardio
labels = ["Normal", "Di atas Normal", "Jauh di atas normal"]
value = df['cholesterol'][df['cardio']==1].value_counts().tolist()
tot = sum(value)
percentages = [(value / tot) * 100 for value in value]
# Create a pie chart with value percentages inside
plt.pie(value, labels=labels, autopct='%.1f%%', startangle=140)
# Draw center circle to make it look like a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title("Cholesterol suffer from cardiovascular disease")
plt.show()

# Plot gluc terhadap cardio
labels = ["Normal", "Di atas Normal", "Jauh di atas normal"]
value = df['gluc'][df['cardio']==1].value_counts().tolist()
tot = sum(value)
percentages = [(value / tot) * 100 for value in value]
# Create a pie chart with value percentages inside
plt.pie(value, labels=labels, autopct='%.1f%%', startangle=140)
# Draw center circle to make it look like a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title("Glucosa suffer from cardiovascular disease")
plt.show()

## Featrue Smoking, Alcohol, & Active

#Create df groupby
dfsmoke_agg = df.groupby('smoke')['cardio'].value_counts().unstack(fill_value=0)
dfalco_agg = df.groupby('alco')['cardio'].value_counts().unstack(fill_value=0)
dfactive_agg = df.groupby('active')['cardio'].value_counts().unstack(fill_value=0)

# Smoking terhadap cardio
list_noncardio = dfsmoke_agg[0].tolist()
list_cardio = dfsmoke_agg[1].tolist()
x = dfsmoke_agg.index
# Set width to give a space for bar chart
width = 0.25
plt.figure(figsize=(12,8))
bar_noncar = plt.bar(x, list_noncardio, width, color='#468B97')
bar_car = plt.bar(x + width, list_cardio, width, color='#CD1818')
# Set the title, label, and legend
plt.xlabel('Perkokok status')
plt.ylabel("Count cardio vs non cardio")
plt.title("Cardio VS Non Cardio  (Aggregated by Smoking)")
plt.legend( (bar_noncar, bar_car ), ('Non Cardio', 'Cardio'))
# Adjustment plot
plt.grid(True, color = "grey", alpha=0.5, linestyle = "-")
plt.xticks(x+width/2,["Non smoking", "Smoking"], rotation=45)
plt.show()

# Alcohol terhadap cardio
list_noncardio = dfalco_agg[0].tolist()
list_cardio = dfalco_agg[1].tolist()
x = dfalco_agg.index
# Set width to give a space for bar chart
width = 0.25
plt.figure(figsize=(12,8))
bar_noncar = plt.bar(x, list_noncardio, width, color='#468B97')
bar_car = plt.bar(x + width, list_cardio, width, color='#CD1818')
# Set the title, label, and legend
plt.xlabel('Status peminum alcohol')
plt.ylabel("Count cardio vs non cardio")
plt.title("Cardio VS Non Cardio  (Aggregated by Alcoholic)")
plt.legend( (bar_noncar, bar_car ), ('Non Cardio', 'Cardio'))
# Adjustment plot
plt.grid(True, color = "grey", alpha=0.5, linestyle = "-")
plt.xticks(x+width/2,["Non Alcoholic", "Alcoholic"], rotation=45)
plt.show()

# Plot active terhadap cardio
list_noncardio = dfactive_agg[0].tolist()
list_cardio = dfactive_agg[1].tolist()
x = dfactive_agg.index
# Set width to give a space for bar chart
width = 0.25
plt.figure(figsize=(12,8))
bar_noncar = plt.bar(x, list_noncardio, width, color='#468B97')
bar_car = plt.bar(x + width, list_cardio, width, color='#CD1818')
# Set the title, label, and legend
plt.xlabel('Status Olahraga')
plt.ylabel("Count cardio vs non cardio")
plt.title("Cardio VS Non Cardio  (Aggregated by Activity)")
plt.legend( (bar_noncar, bar_car ), ('Non Cardio', 'Cardio'))
# Adjustment plot
plt.grid(True, color = "grey", alpha=0.5, linestyle = "-")
plt.xticks(x+width/2,["Tidak pernah berolahraga", "Pernah berolahraga"], rotation=45)
plt.show()

## BMI & Obesitas ##

#Hitung bmi
def hitung_bmi(tinggi, berat):
    tinggi_m = tinggi/100 # konversi ke m
    bmi = berat/ (tinggi_m**2)
    return bmi
# Applying the function
df['BMI'] = df.apply(lambda x: hitung_bmi(x['height'], x['weight']), axis=1)

# Hitung obes
df['obes'] = df['BMI'].apply(lambda x: 3 if x >= 30 else (2 if 25 < x <= 29.9 else (1 if 18.5 <= x < 24.9 else 0)))

#Plot obes terhadap cardio
labels = ["Kelebihan berat badan", "Obesitas", "Berat badan normal", "Kekurangan berat badan"]
value = df['obes'][df['cardio']==1].value_counts().tolist()
tot = sum(value)
percentages = [(value / tot) * 100 for value in value]
# Create a pie chart with value percentages inside
plt.pie(value, labels=labels, autopct='%.1f%%', startangle=140)
# Draw center circle to make it look like a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title("BMI status suffer from cardiovascular disease")
plt.show()

## Blood Press ##

#Hitung bloodpress
df["bld_pres"] = df.apply(lambda row: 1 if 60 <= row["ap_lo"] <= 80 and 90 <= row["ap_hi"] <= 120
                                  else (2 if row["ap_lo"] > 80 and row["ap_hi"] > 120
                                        else (0 if row["ap_lo"] < 60 and row["ap_hi"] < 90
                                              else None)), axis=1)
# Mengisi nilai NaN dengan nilai tertentu, misalnya 3
df['bld_pres'].fillna(3, inplace=True)

#Plot bld press
labels = ["Tekanan darah normal", "Hipertensi", "Hipertensi sitolik atau hipertensi diastolik"]
value = df['bld_pres'][df['cardio']==1].value_counts().tolist()
tot = sum(value)
percentages = [(value / tot) * 100 for value in value]
# Create a pie chart with value percentages inside
plt.pie(value, labels=labels, autopct='%.1f%%', startangle=140)
# Draw center circle to make it look like a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title("Blood pressure of cardiovascular sufferers")
plt.show()

## Korelasi ##

df['cardio'] = df.pop('cardio')
# Look data corelation of the data
# Set figure plt
plt.figure(figsize=(15,10), dpi=100)
# Create mask
mask = np.triu(np.ones_like(df.corr()))
# Plotting
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
# Set the title
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)

## Data prepare ##

# Seperate X and y label
X = df.drop('cardio', axis=1)
y = df['cardio']

# Split and scalling data
X_train, X_test, y_train, y_test = train_test_split(X , y, shuffle = True, test_size = 0.2, random_state = 42)

# Scaling data
scaler = MinMaxScaler()
scaler.fit(X_train)

# The transofrmation of X
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# See the end of dimensions data
print('Dimensi feature data train =', X_train_scaled.shape)
print('Dimensi target data train =', y_train.shape)
print('Dimensi feature data test =', X_test_scaled.shape)
print('Dimensi target data test =', y_test.shape)

## Modeling ##

# Train Data using several ML algorithm

# Logistic Regression
lrmodel = LogisticRegression()
lrstart_time = time.time()
lrmodel.fit(X_train_scaled, y_train)
lrend_time = time.time()
lrruntime = lrend_time - lrstart_time
y_predLr = lrmodel.predict(X_test_scaled)

# KNN
knnmodel = KNeighborsClassifier()
knnstart_time = time.time()
knnmodel.fit(X_train_scaled, y_train)
knnend_time = time.time()
knnruntime = knnend_time - knnstart_time
y_predKnn = knnmodel.predict(X_test_scaled)

# Random Forest
rfmodel = RandomForestClassifier(random_state=42)
rfstart_time = time.time()
rfmodel.fit(X_train_scaled, y_train)
rfend_time = time.time()
rfruntime = rfend_time - rfstart_time
y_predRf = rfmodel.predict(X_test_scaled)

# XGBoost
xgbmodel = xgb.XGBClassifier(random_state=42)
xgbstart_time = time.time()
xgbmodel.fit(X_train_scaled, y_train)
xgbend_time = time.time()
xgbruntime = xgbend_time - xgbstart_time
y_predXgb = xgbmodel.predict(X_test_scaled)

# Model results
algorithm = ['Logistic Regression','K-Nearest Neighbors','Random Forest','XGBoost']
y_pred = [y_predLr, y_predKnn, y_predRf, y_predXgb]
acc = []
for ypred in y_pred:
    acc.append(accuracy_score(y_test, ypred))
    
runtime = [lrruntime, knnruntime, rfruntime, xgbruntime]
datatraining = {"Algorithm": algorithm, "Accuracy": acc, "Runtime": runtime}
datatraining = pd.DataFrame(datatraining)
datatraining

#Hyper parametric tuning
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(estimator=xgbmodel, param_grid=param_grid, cv=3)
# Perform the grid search on the training data
grid_search.fit(X_train_scaled, y_train)
# Get the best parameters and best estimator from the grid search
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
# Make predictions on the test data using the best estimator
y_pred = best_estimator.predict(X_test_scaled)
# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)

# Training with implementating the parameters
xgbmodel = grid_search.best_estimator_
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
xgbmodel.fit(X_train_scaled, y_train, eval_metric=['logloss'], early_stopping_rounds=10, eval_set=eval_set, verbose=True)
y_predXgb = xgbmodel.predict(X_test_scaled)
predictions = [round(value) for value in y_predXgb]

## Evaluate Model ##

# Logloss
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = xgbmodel.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

# Classification Report
print(classification_report(y_test, y_predXgb))

# Confussion Matrix
matrix = confusion_matrix(y_test, y_predXgb)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(matrix, annot =True, fmt = "d", linewidths = 0.5, ax=ax, cmap='Blues')
plt.title(f"Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'], fontsize=16, rotation=360)
ax.set_yticklabels(['Actual Negative', 'Actual Positive'], fontsize=16, rotation=360)
plt.show()

# ROC
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predXgb)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.figure(figsize=(8,8))
plt.title('ROC - AUC plot')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')