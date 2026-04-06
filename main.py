# Import necessary libraries for data processing and visualization
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf                ####data analysic and visulation
import plotly
from plotly.offline import init_notebook_mode, iplot, plot ####creating interactive data visualizations in Python. Here's what each function does:
import joblib     #####This lets you save your work (like a trained model) 

# Initialize Plotly in offline mode for Jupyter Notebook
init_notebook_mode(connected=True)
cf.go_offline()

# Load the dataset
df = pd.read_csv('Cancer.csv')

# Display the first few rows of the dataset
df.head()

# Remove the "Patient Id" column, as it's not needed for analysis
df.drop(['Patient Id'], axis=1, inplace=True)

# Display dataset after dropping "Patient Id"
df.head()

# Check the distribution of 'Level' column
df['Level']

# Replace 'Medium' with 'High' in the 'Level' column to simplify categories
df['Level'] = df['Level'].replace('Medium', 'High')
# Display dataset after replacing 'Medium' with 'High'
df.head()

# Map 'Level' string values to numeric values for modeling
level_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['Level'] = df['Level'].map(level_mapping)

# Convert 'Level' column to numeric
df['Level'] = pd.to_numeric(df['Level'])

# Check for missing values
df.isnull()

# Check if there are any columns with missing values
df.isnull().any()

# Sum of missing values for each column
df.isnull().sum()

# Visualize missing values as a heatmap using Seaborn
import seaborn as sns
sns.heatmap(df.isnull())

# Plot the distribution of 'Smoking' feature
plt.figure(figsize=(10,5))
sns.countplot(x='Smoking', data=df)

# Plot the distribution of 'ChestPain' feature
plt.figure(figsize=(10,5))
sns.countplot(x='ChestPain', data=df)

# Visualize age distribution in relation to 'ChestPain' with a box plot
plt.figure(figsize=(10,5))
sns.boxplot(x='ChestPain', y='Age', data=df)

# Visualize age distribution in relation to 'Smoking' with a box plot
plt.figure(figsize=(10,5))
sns.boxplot(x='Smoking', y='Age', data=df)

# Count 'Smoking' values grouped by 'Age' and display as a DataFrame
sorted_smokers = df.groupby('Age')['Smoking'].count().to_frame()

# Display DataFrame with background color gradient for 'Smoking' counts
sorted_smokers.style.background_gradient(cmap='Reds')

# Display entire DataFrame with background color gradient for values
df.style.background_gradient(cmap='Reds')

# Plot age and smoking data using Plotly's graph objects
import plotly.graph_objects as go

# Start of Random Forest Classifier model training

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
acc_dict = {}

# Separate features and target variable
X = df.drop('Level', axis=1)
y = df['Level']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize and train a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred_randomF = model.predict(X_test)
print('Accuracy score : ', accuracy_score(y_test, y_pred_randomF) * 100)

# Log metrics for Random Forest model
y_pred_proba = model.predict_proba(X_test)
acc_dict['RFC_log_loss'] = log_loss(y_test, y_pred_proba)
acc_dict['RFC_FF1_Score'] = f1_score(y_test, y_pred_randomF, average='weighted')

# Visualize confusion matrix with a heatmap
plt.imshow(np.log(confusion_matrix(y_test, y_pred_randomF)), cmap='Blues', interpolation='nearest')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Start of K-Nearest Neighbors Classifier model training

from sklearn.neighbors import KNeighborsClassifier
score = 0
scores, highscore, bestk = 0, 0, 0

# Find the best 'k' value for KNN model
for k in range(3, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train)
    score = scores.mean()
    if score > highscore:
        highscore = score
        bestk = k
print('Best k is {} with score {}'.format(bestk, highscore))

# Train KNN with the best 'k' value and evaluate
knn = KNeighborsClassifier(n_neighbors=bestk)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('Accuracy score : ', accuracy_score(y_test, y_predict) * 100)
acc_dict['KNN_log_loss'] = log_loss(y_test, y_predict)
acc_dict['KNN_F!1_Score'] = f1_score(y_test, y_predict, average='weighted')

# Visualize KNN confusion matrix with a heatmap
plt.imshow(np.log(confusion_matrix(y_test, y_predict)), cmap='Blues', interpolation='nearest')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Start of K-Means Clustering model training

from sklearn.cluster import KMeans
clf = KMeans()
clf.fit(X_train)
maxx = clf.predict(X_test)
print('Accuracy score : ', accuracy_score(y_test, maxx) * 100)
acc_dict['kMeans_log_loss'] = log_loss(y_test, maxx)
acc_dict['kMeans_F1_Score'] = f1_score(y_test, maxx, average='weighted')

# Visualize KMeans confusion matrix with a heatmap
plt.imshow(np.log(confusion_matrix(y_test, maxx)), cmap='Reds', interpolation='nearest')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Start of Decision Tree Classifier model training

from sklearn.tree import DecisionTreeClassifier
tree_ = DecisionTreeClassifier()
tree_.fit(X_train, y_train)
y_pred = tree_.predict(X_test)
print('Accuracy score : ', accuracy_score(y_test, y_pred) * 100)
acc_dict['Tree_log_loss'] = log_loss(y_test, y_pred)
acc_dict['Tree_f!1_score'] = f1_score(y_test, y_pred)

# Visualize Decision Tree confusion matrix with a heatmap
plt.imshow(np.log(confusion_matrix(y_test, y_pred)), cmap='Blues', interpolation='nearest')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Start of Support Vector Machine (SVM) model training

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy score : ', accuracy_score(y_test, y_pred) * 100)
acc_dict['svc_log_loss'] = log_loss(y_test, y_pred)
acc_dict['svc_f!1_score'] = f1_score(y_test, y_pred)

# Visualize SVM confusion matrix with a heatmap
plt.imshow(np.log(confusion_matrix(y_test, y_pred)), cmap='Blues', interpolation='nearest')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Save the best model (K-Nearest Neighbors) using joblib
joblib.dump(model, "lungs.pkl")
print("Model saved as 'lungs.pkl'")
