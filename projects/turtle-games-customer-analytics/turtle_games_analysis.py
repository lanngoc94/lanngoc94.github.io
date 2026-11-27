# Turtle Games: customer loyalty, segmentation and review sentiment
# Author: Lan Ngoc Mau
# Tools: Python (pandas, statsmodels, scikit-learn, NLTK, TextBlob, VADER, WordCloud)
#
# Business question: how do customers accumulate loyalty points, which customer
# segments should marketing target, and what do product reviews say about the
# customer experience?
#   Part 1  OLS linear regression: loyalty points vs age, income and spending score
#   Part 2  Decision tree regression, pruned and compared across feature sets
#   Part 3  k-means clustering on income and spending score (Elbow + Silhouette)
#   Part 4  NLP: word frequency, word clouds, TextBlob and VADER sentiment
#
# Data files expected in the working directory (not included in this repo):
#   turtle_reviews.csv (Part 1 saves a cleaned copy, turtle_reviews_new.csv)
#
# Exported from a Jupyter notebook: lines starting with "# " were markdown cells.

# # Part 1: How do customers accumulate loyalty points? (OLS linear regression)
# ## 1. Load and explore the data

# Requires: pip install statsmodels

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols

# Linear regression
import sklearn
from sklearn import datasets
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Note: indicates situations that aren't necessarily exceptions.
import warnings
warnings.filterwarnings('ignore')

# Load the CSV file(s) as reviews.

reviews = pd.read_csv('turtle_reviews.csv')
# View the DataFrame.

reviews.info()
reviews

# Any missing values?
reviews.isnull().sum()

# Explore the data.
# Basic descriptive statistics.
reviews.describe()

# Understanding the correlation between only numeric only column.
reviews.corr(numeric_only=True)

# #### Findings from above correlations
# 1. Loyalty points are largely affected by high salary and high spending score.
# 2. Age has almost no correlation with almost everything.
# #### -> These need to be tested

# ## 2. Drop columns

# Drop unnecessary columns.
reviews_new=reviews.drop(columns=['language', 'platform'])

# View column names.
reviews_new

# ## 3. Rename columns

# Rename the column headers.
reviews_new.rename(columns={'remuneration (k£)':'annual_income (k£)',
                            'spending_score (1-100)':'customer_score (1-100)'}, inplace=True)

# View column names.
reviews_new

# ## 4. Save the DataFrame as a CSV file

# Create a CSV file as output
reviews_new.to_csv('turtle_reviews_new.csv', index=False)

# Import new CSV file with Pandas.
turtle= pd.read_csv('turtle_reviews_new.csv')

# View DataFrame.
turtle.info()
turtle

# ## 5. Linear regression

# ### 5a) spending vs loyalty

# Define independent variable.
x1 = turtle['customer_score (1-100)'].values.reshape(-1, 1)

# Define dependent variable.
y1 = turtle['loyalty_points']

# Create model and print summary of metrics.
# Run the OLS test
f = 'y1 ~ x1'

test1 = ols(f, data=turtle).fit()

# View the output

test1.summary()

# Check for linearity
plt.scatter(x1,y1)

# Extract the estimated parameters.
print("R-squared value: ", test1.params)

# Extract the standard errors.
print("Standard errors: ", test1.bse)

# Extract the predicted values.
print("Predicted values: ", test1.predict())

# Set the X coefficient and the constant to generate the regression table.
# Set the coefficient to 33.061693 and the constant to -75.052663 to generate the regression table 
y1_pred = (-75.052663) + 33.061693 * turtle['customer_score (1-100)']
# View the output.
y1_pred

# Plot the data points with a scatterplot

plt.scatter(x1,y1)

# Plot the graph with the regression line (in black)
plt.plot(x1, y1_pred, color='black')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# Adding more information to the plot
plt.xlabel('Customer_Score', fontsize=14, fontweight='bold')
plt.ylabel('Loyalty_Points', fontsize=14, fontweight='bold')
plt.title('Customer Score vs Loyalty Points', fontsize= 18, fontweight='bold')

# View the plot
plt.savefig('customer_score_vs_loyalty_points.png', dpi=300)
plt.show()

# ### 5b) remuneration vs loyalty

turtle

# Define independent variable.

x2 = turtle['annual_income (k£)']
# Define dependent variable.

y2 = turtle['loyalty_points']
# Create model and print summary of metrics.
# Run the OLS test
f2 = 'y2 ~ x2'

test2 = ols(f2, data=turtle).fit()

# View the output

test2.summary()

# Extract the estimated parameters.
print("Parameters: ", test2.params)

# Extract the standard errors.
print("Standard errors: ", test2.bse)

# Extract the predicted values.
print("Predicted values: ", test2.predict())

# Set the the X coefficient and the constant to generate the regression table.
# Set the coefficient to 34.187825  and the constant to -65.686513 to generate the regression table 
y2_pred = (-65.686513) + 34.187825 * turtle['annual_income (k£)']
# View the output.
y2_pred

# Plot the data points with a scatterplot

plt.scatter(x2,y2)

# Plot the graph with the regression line (in orange)
plt.plot(x2, y2_pred, color='orange')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# Adding more information to the plot
plt.xlabel('Annual Income (k£)', fontsize=14, fontweight='bold')
plt.ylabel('Loyalty_Points', fontsize=14, fontweight='bold')
plt.title('Annual Income vs Loyalty Points', fontsize= 18, fontweight='bold')

# View the plot
plt.savefig('annual_income_vs_loyalty_points.png', dpi=300)
plt.show()

# ### 5c) age vs loyalty

# Define independent variable.

x3 = turtle['age']
# Define dependent variable.

y3 = turtle['loyalty_points']
# Create model and print summary of metrics.
# Create formula and pass through OLS methods.

f3 = 'y3 ~ x3'
test3 = ols(f3, data= turtle).fit()

# Print the regression table
test3.summary()

# Extract the estimated parameters.
print("Parameters: ", test3.params)

# Extract the standard errors.

print("Standard errors: ", test3.bse)
# Extract the predicted values.
print("Predicted values: ", test3.predict())

# Set the X coefficient and the constant to generate the regression table
# Set the coefficient to -4.012805  and the constant to 1736.517739 to generate the regression table 
y3_pred = 1736.517739 - 4.012805 * turtle['age']
# View the output.
y3_pred

# Plot the data points with a scatterplot

plt.scatter(x3,y3)

# Plot the graph with the regression line (in orange)
plt.plot(x3, y3_pred, color='red')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# Adding more information to the plot
plt.xlabel('Age (k£)', fontsize=14, fontweight='bold')
plt.ylabel('Loyalty_Points', fontsize=14, fontweight='bold')
plt.title('Age vs Loyalty Points', fontsize= 18, fontweight='bold')

# View the plot
plt.savefig('age_vs_loyalty_points.png', dpi=300)
plt.show()

# ## 6. Observations and insights

# #### Customer Score vs Loyalty Points:
# - 45% of the total variability of loyalty points is explained by customer's spending nature and behaviour.
# - Strong positive relationship.
# - The upward trend is consistent up to a customer score of 60. When the customer score is more than 70, it is shown as the loyalty points they accumulated expands sharply -> this could be understood adding more variables to understand.
# - 2 visible groups: 1 group with high customer score and high loyalty points but also 1 group with low customer score yet low loyalty points.
# #### Annual Income vs Loyalty Points:
# - 38 % of the total variability of loyalty points is explained by annual income of customers.
# - Positive correlation.
# - When earning more than 60K, customers split into 2 groups: 1 group with high accumulated loyalty points and 1 group with low loyalty points
# -> High income does not fully reflect loyalty engagement.
# #### Age vs Loyalty Points:
# Age barely got any impact or correlation to loyalty points.
# -> No predictive power
# #### What could be explored further and suggested future actions.
# 1. As customer score shows a strong relationship towards loyalty points, I would combine this aspect with other aspects such as annual income in order to build a more effective model.
# 2. Understanding why there are a group of customer showing high customer score and high income yet low loyalty points -> is there any reason behind this pattern?
# To be more specific:
# - 55% total variability of loyalty points unexplained by customer score
# - 62% total variability of loyalty points unexplained by annual income
# -> Many other unseen factors might affect overall loyalty points.
# 4. Consider removing age from the model as this factor might not be beneficial in optimizing the model.
# 5. Customer segmentation can be considered

# # Part 2: Decision tree regression
# ## 1. Load and prepare the data

# Import all the necessary packages
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt

# Settings for the notebook.
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = [15, 10]

# Create your new DataFrame.
df = pd.read_csv('turtle_reviews_new.csv')

# View the DataFrame
df

df['gender'].unique()

# ### Transform categorical string values to numeric values for modelling

# #### Gender column
# Decided to make changes in the 'gender' column to change from current string values to numeric values for modelling:
# + 'Male' replaced by 1
# + 'Female' replaced by 0

# Update the gender column with male = 1, and female = 0
df['gender'] = df['gender'].map({'Male':1, 'Female':0})

# View the output
df

# #### Education column
# The education levels follow a natural low-to-high order, decided to change from string values to numeric values. In particular:
# + "Basic" : 0
# + "Diploma": 1
# + Graduate": 2
# + "Postgraduate": 3
# + "PhD": 4

df['education'].value_counts()

df['education'].unique().tolist()

# Plot graph for individual columns to understand the distribution
# Gender male = 1, and female = 0 
plt.rcParams['figure.figsize'] = [6, 3]
sns.countplot(x=df['gender'])
plt.show()
sns.countplot(x=df['education'])
plt.show()

# Update the education column: "Basic" = 0, "Diploma" = 1, "Graduate" = 2, "Postgraduate" = 3, "PhD" = 4
df['education'] = df['education'].map({'Basic':0, 'diploma':1,
                                       'graduate':2,'postgraduate':3, 'PhD':4})

# View the output
df

df.columns.tolist()

# Specify Y.
y = df['loyalty_points']
# Specify X.
cols = ['gender','age','annual_income (k£)','customer_score (1-100)']
# cols1 = ['age','annual_income (k£)','customer_score (1-100)']
X = df[cols]

# Review X and Y.
X.shape, y.shape

# ## 2. Create train and test data sets.

# Split the data into test and train data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# ## 3. Create Decision tree regressor

# Create your decision tree regressor.
from sklearn import tree
my_tree = DecisionTreeRegressor(random_state=42, max_depth=7)

# Train the Decision Tree Regressor
reg = my_tree.fit(X_train, y_train)

# Predict the response for the dataset
y_pred = reg.predict(X_test)

# Scatter plot for predicted and tested value
plt.scatter(y_test, y_pred)
plt.xlabel("True loyalty points from test set")
plt.ylabel("Predicted prices from trained model")
plt.title(" Predicted vs Tested Royalty Points")
plt.savefig('decision_tree_regression_1.png', dpi=300)
plt.show()

# Getting mean absolute error
from sklearn.metrics import mean_absolute_error as MAE
err = MAE(y_test, y_pred)
print("Mean absolute error of test set is: ", err)

# Plot the tree
plt.figure(figsize=(40, 20)) 
tree.plot_tree(reg, filled=True)
plt.savefig('decision_tree_regressor_2.png', dpi=800)

# #### Evaluate the model (MAE, MSE, RMSE and R²)

accuracy = pd.DataFrame({'Actual Value': y_test, 'Predicted Values': y_pred})  
print(accuracy)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.4f}')
print(f'R² as a percentage: {r2 * 100:.2f}%')

# ### Scenario 2: Test using variables with 3 columns only: age, customer_score, annual income

# Specify y.
y1 = df['loyalty_points']
# Specify X.
cols1 = ['age','annual_income (k£)','customer_score (1-100)']
X1 = df[cols1]

print(X1)
print(y1)

# Split the data into test and train data.
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size= 0.3, random_state= 42)

X1_train.shape, y1_train.shape, X1_test.shape, y1_test.shape

# Create your decision tree regressor.
from sklearn import tree
my_tree_1 = DecisionTreeRegressor(random_state=42, max_depth=7)

# Train the Decision Tree Regressor
reg_1 = my_tree_1.fit(X1_train, y1_train)

# Predict the response for the dataset
y1_pred = reg_1.predict(X1_test)

# Scatter plot for predicted and tested value
plt.scatter(y1_test, y1_pred)
plt.xlabel("True loyalty points from test set")
plt.ylabel("Predicted prices from trained model")
plt.title(" Predicted vs Tested Royalty Points")
plt.savefig('decision_tree_regression_3.png', dpi=300)
plt.show()

# Plot the tree
plt.figure(figsize=(40, 20)) 
tree.plot_tree(reg_1, filled=True)
plt.savefig('decision_tree_regressor_4.png', dpi=800)

accuracy_1 = pd.DataFrame({'Actual Value': y1_test, 'Predicted Values': y1_pred})  
print(accuracy_1)

print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_pred)
print(f'R² Score: {r2_1:.4f}')
print(f'R² as a percentage: {r2_1 * 100:.2f}%')

# ### Scenario 3: Test using variables with 2 columns only: customer_score, annual income

# Specify y.
y2 = df['loyalty_points']
# Specify X.
cols2 = ['annual_income (k£)','customer_score (1-100)']
X2 = df[cols2]

print(X2)
print(y2)

# Split the data into test and train data.
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size= 0.3, random_state= 42)

X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape

# Create your decision tree regressor.
from sklearn import tree
my_tree_2 = DecisionTreeRegressor(random_state=42, max_depth=7)

# Train the Decision Tree Regressor
reg_2 = my_tree_2.fit(X2_train, y2_train)

# Predict the response for the dataset
y2_pred = reg_2.predict(X2_test)

# Scatter plot for predicted and tested value
plt.scatter(y2_test, y2_pred)
plt.xlabel("True loyalty points from test set")
plt.ylabel("Predicted prices from trained model")
plt.title(" Predicted vs Tested Royalty Points")
plt.savefig('decision_tree_regression_5.png', dpi=300)
plt.show()

# Plot the tree
plt.figure(figsize=(40,20)) 
tree.plot_tree(reg_2, filled=True)
plt.savefig('decision_tree_regressor_6.png', dpi=800)

accuracy_2 = pd.DataFrame({'Actual Value': y2_test, 'Predicted Values': y2_pred})  
print(accuracy_2)

print('Mean Absolute Error:', metrics.mean_absolute_error(y2_test, y2_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y2_test, y2_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2_test, y2_pred)))

from sklearn.metrics import r2_score

r2_2 = r2_score(y2_test, y2_pred)
print(f'R² Score: {r2_2:.4f}')
print(f'R² as a percentage: {r2_2 * 100:.2f}%')

# #### Interpret model and note observations.
# - Results when testing with 4 columns: age, gender, customer_Score, income, max_depth=7
# R²: 97.74%
# - Result when testing with 3 columns: age, customer_score, income, max_depth = 7
# R²: 97.73%
# - Result when testing with 2 columns: customer_score, income, max_depth = 7
# R²: 97.02%
# The depth of 7 gives a high R², will try with different max depth length later to decide how to prune the tree

# ## 4. Fit and plot final model.

# Prune the model.
# Determine the optimal value for max depth, then prune the decision tree
# One approach often applied is to look at a range of values for the parameter under investigation

# List of values to try for max_depth:
max_depth_range = list(range(1, 7))

# have a loop that fits decision trees for different 'max_depth', and calculate/store the R² of each tree
accuracy_3 = []
for depth in max_depth_range:    
    reg_3 = DecisionTreeRegressor(max_depth = depth, random_state = 0)
    reg_3.fit(X_train, y_train)
    score = reg_3.score(X_test, y_test)
    accuracy_3.append(score)
    
# Plot R² values across the range of depth values evaluated
plt.rcParams['figure.figsize'] = [10, 5]
accuracy_3 = pd.DataFrame(accuracy_3)
accuracy_3.plot()
plt.show()

# Split the data into test and train data.
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size= 0.3, random_state= 42)

X_train_final.shape, y_train_final.shape, X_test_final.shape, y_test_final.shape

# Fit and plot final model.
# Create your decision tree regressor.
from sklearn import tree
my_tree_final= DecisionTreeRegressor(random_state=42, max_depth=3)

# Train the Decision Tree Regressor
reg_final = my_tree_final.fit(X_train_final, y_train_final)

# Predict the response for the dataset
y_pred_final = reg.predict(X_test_final)

# Scatter plot for predicted and tested value
plt.scatter(y_test_final, y_pred_final)
plt.xlabel("True loyalty points from test set")
plt.ylabel("Predicted prices from trained model")
plt.title(" Predicted vs Tested Royalty Points")
plt.savefig('decision_tree_regression_3.png', dpi=300)
plt.show()

# Plot the tree
plt.figure(figsize=(40,20)) 
tree.plot_tree(reg_final, feature_names=cols, filled=True)
plt.savefig('decision_tree_regressor_final.png', dpi=800)

print("Tree depth =", reg_final.get_depth(),'\n'
      "Number of leaves =", reg_final.get_n_leaves())

from sklearn.metrics import r2_score

r2_final = r2_score(y_test_final, y_pred_final)
print(f'R² Score: {r2_final:.4f}')
print(f'R² as a percentage: {r2_final * 100:.2f}%')

# Finding out about Feature importance to help in the decision making process
reg_input = reg_final
importances = pd.DataFrame({'feature':X_train_final.columns,'importance':np.round(reg_input.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
importances

accuracy_final = pd.DataFrame({'Actual Value': y_test_final, 'Predicted Values': y_pred_final})  
print(accuracy_final)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_final, y_pred_final))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_final, y_pred_final))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_final, y_pred_final)))

# ## 5. Insights and observations

# #### Interpret model and note observations.
# At first, I studied all variables possible: gender, age, customer score and, annual income. However, by plotting the trees using different variables combinations, seeing statistical results, and understanding the importance of each variable in the tree model, my findings would be:
# - Only 2 features actually have an impact on loyalty points: annual income and customer score:
# + annual income slightly affects more than the other: 52.5%
# + customer score: 47.5%
# - Gender and age are irrelevant in this case as they showed zero predictive power, the result of the predictive model with or without their presence remains unchanged
# -> We should be targeting to understand how customer score and annual income might affect loyalty points, and demographics such as gender and age play no important role in defining loyalty points.

# ### Decision Tree Analysis
# - The tree decision making process was based entirely on customer score and annual income, no gender or age was used as they do not have predictive power.
# - Value: loyalty points prediction
# - Root node: customer score <=67, 1400 training samples split into 982 low score (<=67), and 418 high score (>67)
# + customer score above 67: loyalty points 2767
# + customer score less than 67: loyalty points 1081
# -> Customers with higher scores earn on average 2.5 times more loyalty points than those with scores less than 67.
# - Highest score customer(customer score above 67) with highest income (more than 74k) earns around 4,849 points: this can be considered the most valuable customer segment.
# - The pruned tree (max depth 3) splits customers into 8 groups based on a combination of income and customer score.
# -> Suggestion: any loyalty program should focus on encouraging customer to increase their scores and targeting on higher income brackets.

# #

# # Part 3: Customer segmentation with k-means clustering
# ## 1. Load and explore the data

# Import necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')

# Load the CSV file(s) as df2.
df2= pd.read_csv('turtle_reviews_new.csv')

# View DataFrame.
df2
df2.columns.tolist()

df2

# Drop unnecessary columns.
df3 = df2[['annual_income (k£)','customer_score (1-100)','gender', 'education']]
# View DataFrame.
df3

# Explore the data.
df3.info()

# Descriptive statistics.
df3.describe()

# ## 2. Plot

# Create a scatterplot with Seaborn, using gender=hue

sns.scatterplot(x='annual_income (k£)', y='customer_score (1-100)', hue='gender',data=df3)

# Create a scatterplot with Seaborn, using gender=education

sns.scatterplot(x='annual_income (k£)', y='customer_score (1-100)', hue='education',data=df3)

# Create a pairplot with Seaborn using hue='gender'

x= df3[['annual_income (k£)', 'customer_score (1-100)']]
sns.pairplot(df3, vars=x, hue='gender', diag_kind='kde')

# Create a pairplot with Seaborn using hue='education'

x= df3[['annual_income (k£)', 'customer_score (1-100)']]
sns.pairplot(df3, vars=x, hue='education', diag_kind='kde')

# ## 3. Elbow and silhouette methods

# Determine the number of clusters: Elbow method.
ss= []
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,
                   init='k-means++',
                   max_iter=300,
                   n_init=10,
                   random_state=0)
    kmeans.fit(x)
    ss.append(kmeans.inertia_)

# Plot the elbow method: Plot the sum of squared distances from each point to its assigned centre

plt.plot(range(1,11), ss, marker='o')

# Insert labels and title
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("SS Distance")
plt.savefig('elbow_method.png',dpi=300)
plt.show()

# Determine the number of clusters: Silhouette method.
# Find the range of clusters to be used using silhouette method
sil=[]
kmax=10

for k in range(2, kmax+1):
    kmeans_s = KMeans(n_clusters=k).fit(x)
    labels= kmeans_s.labels_
    sil.append(silhouette_score(x, labels, metric='euclidean'))

# Plot the silhouette method.
plt.plot(range(2, kmax+1), sil, marker='o')

# Insert the labels and title.
plt.title("The Silhouette Method")
plt.xlabel("Number of clusters")
plt.ylabel("Sil")
plt.savefig('silouehette_method.png',dpi=300)
plt.show()

# #### Findings:
# - Elbow method: curve started to flatten at k=5
# - Silhouette method: peaks at 5
# -> 5 might be the optimal of clusters for current data
# -> Customers are divided into 5 different group segments based on annual income and customer score

# ## 4. Evaluate k-means model at different values of *k*

# #### Five clusters

# Use five clusters as per 
kmeans = KMeans(n_clusters = 5, 
               max_iter=15000,
               init='k-means++', 
               random_state=0).fit(x)
clusters= kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x, hue='K-Means Predicted', diag_kind='kde')

# Check the number of observations per predicted class
x['K-Means Predicted'].value_counts()

# View the K-Means predicted
print(x.head())

# Visualising the clusters.
# Set the plot size
sns.set(rc = {'figure.figsize':(12, 8)})

# Create a scatterplot
sns.scatterplot(x='annual_income (k£)', y='customer_score (1-100)', 
                data=x, 
                hue='K-Means Predicted',
                palette=['red', 'green', 'blue','orange', 'black'])

# #### Four clusters

# Use five clusters as per 
x4= df3[['annual_income (k£)', 'customer_score (1-100)']]
kmeans_4 = KMeans(n_clusters = 4, 
               max_iter=15000,
               init='k-means++', 
               random_state=0).fit(x4)
clusters_4= kmeans_4.labels_
x4['K-Means Predicted'] = clusters_4

# Plot the predicted.
sns.pairplot(x4, hue='K-Means Predicted', diag_kind='kde')

# Check the number of observations per predicted class
x4['K-Means Predicted'].value_counts()

# View the K-Means predicted.
print(x4.head())

# Visualising the clusters.
# Set the plot size
sns.set(rc = {'figure.figsize':(12, 8)})

# Create a scatterplot
sns.scatterplot(x='annual_income (k£)', y='customer_score (1-100)', 
                data=x4, 
                hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'black'])

# #### Decision about choosing which number of clusters to go:

# The number of customers in groups 2, 1 and 3 is similar when using k=5 or k=4
# When using k=5, the group 0 in k=4 were divided to group 0 and group 4. Creating 2 new distinct groups clearly defined the distinction between each customers groups.
# -> Final decision: choosing k=5 -> divide customers into 5 groups

# ## 5. Fit final model and justify your choice

# Use five clusters as per 
kmeans = KMeans(n_clusters = 5, 
               max_iter=15000,
               init='k-means++', 
               random_state=0).fit(x)
clusters= kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x, hue='K-Means Predicted', diag_kind='kde')

# Check the number of observations per predicted class
x['K-Means Predicted'].value_counts()

# ## 6. Plot and interpret the clusters

# Visualising the clusters.
# Set the plot size
sns.set(rc = {'figure.figsize':(12, 8)})

# Create a scatterplot
sns.scatterplot(x='annual_income (k£)', y='customer_score (1-100)', 
                data=x, 
                hue='K-Means Predicted',
                palette=['red', 'green', 'blue','orange', 'black'])

# View the K-Means predicted
plt.savefig('scatterplot_kmeans_5.png', dpi=300)
print(x.head())

# ## 7. Insights and observations

# > By using k-means clusterings methods, I have managed to identify different customer groups with similar customer behaviour based on their annual income and customer score.
# > In particular:
# - Cluster 0 — Middle Income, Average Score (Red, n=774 — largest group)
# Income: ~£40–65k, Score: ~40–60
# The "average" customer — moderate spender, moderate engagement
# - Cluster 2 — High Income, High Score (Blue, n=356)
# Income: ~£55–100k+, Score: ~70–97
# - Cluster 1 — High Income, Low Score (Green, n=330)
# Income: ~£55–105k+, Score: ~1–40
# Earn well but disengage — high potential, worth deep dive more
# - Cluster 4 — Low Income, Low Score (Black, n=271)
# Income: ~£10–35k, Score: ~3–40
# Least engaged and lowest spending
# - Cluster 3 — Low Income, High Score (Orange, n=269)
# Income: ~£10–35k, Score: ~65–99
# Loyal despite low income — highly engaged, likely value-driven customers

# #

# # Part 4: Sentiment analysis of customer reviews (NLP)
# ## 1. Load and explore the data

# Import all the necessary packages.
import pandas as pd
import numpy as np
import nltk 
import os 
import matplotlib.pyplot as plt

# Import the required tokenisation model
nltk.download ('punkt')
nltk.download ('stopwords')

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy.stats import norm

# Import Counter.
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# View DataFrame.
reviews_new

# Explore data set.
reviews_new.isnull().sum()

# Keep necessary columns. Drop unnecessary columns.
survey = reviews_new[['review', 'summary']]

# View DataFrame.
survey

# Determine if there are any missing values.
survey.isnull().sum()

# ## 2. Prepare the data for NLP
# ### 2a) Change to lower case and join the elements in each of the columns respectively (review and summary)

# Review: Change all to lower case and join with a space.
survey['review'] = survey['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Summary: Change all to lower case and join with a space.
survey['summary'] = survey['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# View the DataFrame after the change
survey

# ### 2b) Replace punctuation in each of the columns respectively (review and summary)

# Replace all the punctuation in the review column.
survey['review'] = survey['review'].str.replace('[^\w\s]', '', regex=True)

# View output.
survey['review']

# Replace all the punctuation in the summary column.
survey['summary'] = survey['summary'].str.replace('[^\w\s]', '', regex=True)

# View output.
survey['summary']

# ### 2c) Drop duplicates in both columns

# Drop duplicates in both columns.
survey = survey.drop_duplicates(subset=['review', 'summary'])
# View DataFrame.
survey

# ## 3. Tokenise and create wordclouds

# Create new DataFrame (copy DataFrame).

survey1 = survey
# View DataFrame.
survey1

# Install the required tokenisation function.
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Apply tokenisation to both columns.
survey1['review'] = survey1['review'].apply(word_tokenize)
survey1['summary']= survey1['summary'].apply(word_tokenize)

# View DataFrame.
survey1.reset_index(inplace=True)
survey1

# ### Analyse review column

# Define an empty list of tokens.
all_tokens_review = []

for i in range(survey1.shape[0]):
    # Add each token to the list.
    all_tokens_review = all_tokens_review + survey1['review'][i]

# Import the FreqDist class.

from nltk.probability import FreqDist

# Calculate the frequency distribution

fdist_review = FreqDist(all_tokens_review)

# Preview the data
fdist_review

# Filter out tokens that are neither alphabets nor numbers (to eliminate punctuation marks, etc.).
tokens_review = [word for word in all_tokens_review if word.isalnum()]

# #### Eliminate stopwords in review column

# Download the stopword list.
nltk.download ('stopwords')
from nltk.corpus import stopwords

# Create a set of English stopwords.
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of tokens without stopwords.
tokens_review_2 = [x for x in tokens_review if x.lower() not in english_stopwords]

# Define an empty string variable.
tokens2_string_review = ''

for value in tokens_review_2:
    # Add each filtered token word to the string.
    tokens2_string_review = tokens2_string_review + value + ' '

# #### Create a word cloud

wordcloud = WordCloud(width = 1600, height = 900, 
                background_color ='white', 
                colormap='plasma', 
                min_font_size = 10).generate(tokens2_string_review) 

# Plot the WordCloud image.                        
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.savefig('wordcloud_review.png', dpi=300)
plt.show()

# ### Analyse Summary column

# Define an empty list of tokens.
all_tokens_summary = []

for i in range(survey1.shape[0]):
    # Add each token to the list.
    all_tokens_summary = all_tokens_summary + survey1['summary'][i]

# Import the FreqDist class.

from nltk.probability import FreqDist

# Calculate the frequency distribution

fdist_summary = FreqDist(all_tokens_summary)

# Preview the data
fdist_summary

# Filter out tokens that are neither alphabets nor numbers (to eliminate punctuation marks, etc.).
summary_review = [word for word in all_tokens_summary if word.isalnum()]

# #### Eliminate stopwords in the summary column

# Download the stopword list.
nltk.download ('stopwords')
from nltk.corpus import stopwords

# Create a set of English stopwords.
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of tokens without stopwords.
summary_review_2 = [x for x in summary_review if x.lower() not in english_stopwords]

# Define an empty string variable.
tokens2_string_summary = ''

for value in summary_review_2:
    # Add each filtered token word to the string.
    tokens2_string_summary = tokens2_string_summary + value + ' '

# #### Create a word cloud

wordcloud = WordCloud(width = 1600, height = 900, 
                background_color ='white', 
                colormap='plasma', 
                min_font_size = 10).generate(tokens2_string_summary) 

# Plot the WordCloud image.                        
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.savefig('wordcloud_summary.png', dpi=300)
plt.show()

# ## 4. Frequency distribution and polarity
# ### 4a) Create frequency distribution for both columns

# Determine the frequency distribution.
# Define an empty list of tokens.
all_tokens_final = []

for i in range(survey1.shape[0]):
    # Add each token to the list.
    all_tokens_final  = all_tokens_summary + all_tokens_review

print(all_tokens_final)

# Import the FreqDist class.

from nltk.probability import FreqDist

# Calculate the frequency distribution

fdist_final = FreqDist(all_tokens_final)

# Preview the data
fdist_final

# ### 4b) Remove alphanumeric characters and stopwords

# Filter out tokens that are neither alphabets nor numbers (to eliminate punctuation marks, etc.).
tokens_final = [word for word in all_tokens_final if word.isalnum()]

# Remove all the stopwords

nltk.download ('stopwords')
from nltk.corpus import stopwords

# Create a set of English stopwords.
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of tokens without stopwords.
tokens_final_2 = [x for x in tokens_final if x.lower() not in english_stopwords]

# Define an empty string variable.
tokens2_final = ''

for value in tokens_final_2:
    # Add each filtered token word to the string.
    tokens2_final = tokens2_final + value + ' '

# ### 4c) Create wordcloud without stopwords

# Create a wordcloud without stop words.
wordcloud = WordCloud(width = 1600, height = 900, 
                background_color ='white', 
                colormap='plasma', 
                min_font_size = 10).generate(tokens2_final) 

# Plot the WordCloud image.                        
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.savefig('wordcloud_combined_summary_review.png', dpi=300)
plt.show()

# ### 4d) Identify 15 most common words and polarity

# Determine the 15 most common words in review column
fdist_review = FreqDist(tokens_review_2)
fdist_review

# Import the Counter class.
from collections import Counter

# Generate a DataFrame from Counter.
counts_review = pd.DataFrame(Counter(tokens_review_2).most_common(15),
                      columns=['Word', 'Frequency']).set_index('Word')

# Preview data.
counts_review

# Determine the 15 most common words in the summary column
fdist_summary = FreqDist(summary_review_2)
fdist_summary

# Import the Counter class.
from collections import Counter

# Generate a DataFrame from Counter.
counts_summary = pd.DataFrame(Counter(summary_review_2).most_common(15),
                      columns=['Word', 'Frequency']).set_index('Word')

# Preview data.
counts_summary

# Determine the 15 most common words in both columns
fdist_final = FreqDist(tokens_final_2)
fdist_final

# Import the Counter class.
from collections import Counter

# Generate a DataFrame from Counter.
counts_final = pd.DataFrame(Counter(tokens_final_2).most_common(15),
                      columns=['Word', 'Frequency']).set_index('Word')

# Preview data.
counts_final

# ## 5. Review Sentiment Polarity and Subjectivity: Plot histograms of polarity sentiment/ subjectivity (use 15 bins) and sentiment scores/subjectivity scores for the respective columns.

# #### Determine Sentiment Polarity columns

# Install TextBlob.
# Requires: pip install textblob

# Import the necessary package.
from textblob import TextBlob

def generate_polarity(comment):
    '''Extract polarity score (-1 to +1) for each comment'''
    return TextBlob(comment).sentiment[0]

# Determine a new DataFrame
survey2 = survey1.copy()

# Rejoin tokens back into strings for both columns
survey2['review'] = survey2['review'].apply(lambda x: ' '.join(x))
survey2['summary'] = survey2['summary'].apply(lambda x: ' '.join(x))

# View the DataFrame
survey2

# Determine sentiment polarity of both columns. 
survey2['review_polarity'] = survey2['review'].apply(generate_polarity)
survey2['summary_polarity']= survey2['summary'].apply(generate_polarity)
survey2['combined_polarity'] = (survey2['review_polarity'] + survey2['summary_polarity'])/2
# View output.
survey2

# #### Determine Subjectivity columns

# Define a function to extract a subjectivity score for the comment.
def generate_subjectivity(comment):
    return TextBlob(comment).sentiment[1]

# Determine subjectivity of both columns.  
survey2['review_subjectivity'] = survey2['review'].apply(generate_subjectivity)
survey2['summary_subjectivity']= survey2['summary'].apply(generate_subjectivity)
survey2['combined_subjectivity'] = (survey2['review_subjectivity'] + survey2['summary_subjectivity'])/2
# View output.
survey2

# #### Visualisations with Review columns

# > Using only Review Polarity Column

# Review: Create a histogram plot with bins = 15.
# Histogram of Sentiment Polarity
# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(16,9))

# Define the bars.
n, bins, patches = plt.hist(survey2['review_polarity'], num_bins, facecolor='red', alpha=0.6)

# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of Review Sentiment Polarity', fontsize=20)

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Review Polarity
axes[0].hist(survey2['review_polarity'], bins=15, facecolor='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].set_title('Review', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Polarity Score', fontsize=12,fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[0].axvline(survey2['review_polarity'].mean(), color='#E74C3C', linestyle='-', linewidth=1.5, label='Mean')
axes[0].legend(fontsize=10)

# Summary Polarity
axes[1].hist(survey2['summary_polarity'], bins=15, facecolor='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].set_title('Summary', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Polarity Score', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[1].axvline(survey2['summary_polarity'].mean(), color='#3498DB', linestyle='-', linewidth=1.5, label='Mean')
axes[1].legend(fontsize=10)

# Combined Polarity
axes[2].hist(survey2['combined_polarity'], bins=15, facecolor='#2ECC71', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[2].set_title('Combined', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Polarity Score', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Count', fontsize=12,fontweight='bold' )
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[2].axvline(survey2['combined_polarity'].mean(), color='#2ECC71', linestyle='-', linewidth=1.5, label='Mean')
axes[2].legend(fontsize=10)

plt.suptitle('Sentiment Polarity Scores', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sentiment_polarity_scores.png', dpi=300)
plt.show()

# #### Visualisation with subjectivity columns

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Review Polarity
axes[0].hist(survey2['review_subjectivity'], bins=15, facecolor='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].set_title('Review', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Subjectivity Score', fontsize=12,fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[0].axvline(survey2['review_subjectivity'].mean(), color='#E74C3C', linestyle='-', linewidth=1.5, label='Mean')
axes[0].legend(fontsize=10)

# Summary Polarity
axes[1].hist(survey2['summary_subjectivity'], bins=15, facecolor='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].set_title('Summary', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Subjectivity Score', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[1].axvline(survey2['summary_subjectivity'].mean(), color='#3498DB', linestyle='-', linewidth=1.5, label='Mean')
axes[1].legend(fontsize=10)

# Combined Polarity
axes[2].hist(survey2['combined_subjectivity'], bins=15, facecolor='#2ECC71', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[2].set_title('Combined', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Subjectivity Score', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Count', fontsize=12,fontweight='bold' )
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[2].axvline(survey2['combined_subjectivity'].mean(), color='#2ECC71', linestyle='-', linewidth=1.5, label='Mean')
axes[2].legend(fontsize=10)

plt.suptitle('Subjectivity Scores', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('subjectivity_scores.png', dpi=300)
plt.show()

# ## 6. Identify top 20 positive and negative reviews and summaries respectively

survey2.columns.tolist()

# #### Review columns

# Top 20 negative reviews.

negative_sentiment_review = survey2.nsmallest(20, 'review_polarity')

# Eliminate unnecessary columns.
negative_sentiment_review = negative_sentiment_review[['review',
 'review_polarity',
 'review_subjectivity']]

# Adjust the column width.
negative_sentiment_review.style.set_properties(subset=['review'], **{'width': '1200px'})

# Top 20 positive reviews.

positive_sentiment_review = survey2.nlargest(20, 'review_polarity')

# Eliminate unnecessary columns.
positive_sentiment_review = positive_sentiment_review[['review',
 'review_polarity',
 'review_subjectivity']]

# Adjust the column width.
positive_sentiment_review.style.set_properties(subset=['review'], **{'width': '1200px'})

# #### Summary columns

# Top 20 negative summary.

negative_sentiment_summary = survey2.nsmallest(20, 'summary_polarity')

# Eliminate unnecessary columns.
negative_sentiment_summary= negative_sentiment_summary[['summary',
 'summary_polarity',
 'summary_subjectivity']]

# Adjust the column width.
negative_sentiment_summary.style.set_properties(subset=['summary'], **{'width': '1200px'})

# Top 20 positive summary.

positive_sentiment_summary = survey2.nlargest(20, 'summary_polarity')

# Eliminate unnecessary columns.
positive_sentiment_summary= positive_sentiment_summary[['summary',
 'summary_polarity',
 'summary_subjectivity']]

# Adjust the column width.
positive_sentiment_summary.style.set_properties(subset=['summary'], **{'width': '1200px'})

# ## 7. Insights and observations
# Sentiment polarity scores are assigned on a range, where -1 is the lowest negative sentiment, and +1 is the highest possible positive sentiment.
# Subjectivity ranges from 0 to 1, where 0 means fact-based and objective, while 1 means opinion-based and subjective.

# ### 1. Sentiment Polarity Scores
# Review polarity:
# - Overall reviews are mildly positive.
# - The scores range widely between -1 and 1: mixed opinions across the dataset
# - Most reviews fall between -0.25 and 0.5
# Summary polarity:
# - Heavily concentrated at 0 (neutral)
# - Large spike at 0: this might be explained by very short and single words that TextBlob could not capture precisely.
# -> The result is not entirely reliable as the model might not capture the real picture.
# Combined polarity:
# - combined score leans positive overall
# - mean line sits to the right of neutral -> customer are slightly more positive than negative.
# ### 2. Subjectivity Scores
# Review subjectivity:
# + peaking around 0.4-0.5
# + most reviews are moderately subjective - a mix of opinion and fact
# + very few reviews are fact-based or purely objective (near 0)
# Summary subjectivity:
# + Massive spike at 0 - most summaries are completely objective: this raises a question: can TextBlob detect subjectivity reliably on very short text?
# + The rest of distribution spreads fairly evenly from 0.2 to 1.0
# Combined subjectivity:
# + More balanced and normally distributed
# + Peak around 0.4 - 0.6: moderate subjectivity overall
# ### 3. Top 20 Negative Reviews
# - Difficult instructions: product usability issue: "instructions are complicated", "difficult directions": very common
# - Not age-appropriate: misleading product description: "not for a young child", "8 year old frustrated": common
# - Poor value for money: pricing concern: "expensive for what you get": moderate
# - Product defects: quality control issue: "incomplete kit", "both parts were the same": moderate
# - Gift disappointment:"bought as christmas gift", "gift for daughter": not arrive at expected: common
# Special note: Reviews 421, 490, 795, 798, 814 have clearly positive text ("my son loves this game", "this game is a blast") but show -0.4 polarity — TextBlob is misclassifying these, likely because of negative words like "kids" or sentence structure confusing the model.
# ### 4. Top 20 Positive Reviews
# - Single word reviews: "awesome", "perfect": Very high subjectivity (1.0) but low informational value: TextBlob might give short and positive words with maximum scores -> the positive pictures might be inflated.
# - Gift satisfaction: "my daughter loves her stickers": Strong positive gifting experience
# - Educational value: "excellent activity for teaching selfmanagement": Product seen as educational tool: common
# - Age appropriate products: "perfect for preschooler", "well designed for 9 year olds": common
# - Overall product quality: "came in perfect condition", "wonderful product": very common

# ### 5. Top 20 Negative Summary
# - Not interested: boring", "small and boring", "dumb and boring: Most common complaint
# - Disappointment: "disappointed" appears 4 times: Repeated single-word summaries
# - Quality issues: "bad quality all made of paper": Material quality concern
# - Wrong expectations: "not what I was expecting": Product description mismatch
# - Age/difficulty: "at age 31 found these very difficult": difficulty beyond just children
# ### 6. Top 20 Positive Summary
# - Single words: "perfect", "awesome", "excellent": positive summary but quite short
# - Educational angle: "excellent therapy tool", "best easter teaching tool": Strong educational positioning
# - Age specific praise: "perfect for preschooler", "awesome and well designed for 9 year olds": Age-appropriate products well received
# ### Limitations of Analysis:
# - TextBlob struggles with short text: Summary scores less reliable — single words like "boring" or "awesome" get maximum scores
# - Some misclassifications detected: Positive reviews in rows 421, 490, 795 incorrectly scored as negative (-0.4)
# - Stopword removal may remove context: Words like "not" can be removed, flipping the actual meaning
# - Use other types of sentiment analysis for comparison
# ### Recommendations
# - Improve product instructions:
# more visual step-by-step guides - ideally with images or video links included in the packaging.
# - Review age range labelling:
# Multiple reviewers noted products were not suitable for the age ranges advertised. A thorough review of age recommendations on all product listings is strongly advised to reduce disappointment and returns.
# - Improve quality control:
# Defective products such as incomplete kits and mismatched parts appeared in the most negative reviews. Stronger pre-shipment quality checks would directly reduce the volume of 1-star reviews.
# - Highlight educational value: Reviewers who mentioned therapeutic or educational use were consistently positive. Positioning the product range more explicitly as educational or therapeutic tools could attract a wider, more engaged customer base.

# # Part 4b: VADER sentiment analysis as a cross-check on TextBlob

survey2

# Create a new DataFrame for VaderSentiment analysis
survey3 = survey2[['review', 'summary']]
survey3.style.set_properties(subset=['review', 'summary'], **{'width': '1200px'})

survey3.shape

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialise the analyser
analyzer = SentimentIntensityAnalyzer()

# Test on row 1 — negative review
test1 = "a beautiful and sentimental gift i plan to order 2 more sets for gifts"
print(analyzer.polarity_scores(test1))

# Test on row 3 — positive review
test2 = "amazing buy bought it as a gift for our new dm and its perfect"
print(analyzer.polarity_scores(test2))

# #### Apply VADER to survey3

# Compound score (main metric -1 to 1)
survey3['review_vader_compound'] = survey3['review'].apply(
    lambda x: analyzer.polarity_scores(x)['compound'])

survey3['summary_vader_compound'] = survey3['summary'].apply(
    lambda x: analyzer.polarity_scores(x)['compound'])

# Positive, negative, neutral scores
survey3['review_vader_pos'] = survey3['review'].apply(
    lambda x: analyzer.polarity_scores(x)['pos'])

survey3['review_vader_neg'] = survey3['review'].apply(
    lambda x: analyzer.polarity_scores(x)['neg'])

survey3['review_vader_neu'] = survey3['review'].apply(
    lambda x: analyzer.polarity_scores(x)['neu'])

# Combined compound score
survey3['combined_vader_compound'] = (
    survey3['review_vader_compound'] + survey3['summary_vader_compound']
) / 2

# View result
survey3

def get_vader_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

survey3['review_vader_label'] = survey3['review_vader_compound'].apply(get_vader_label)
survey3['summary_vader_label'] = survey3['summary_vader_compound'].apply(get_vader_label)
survey3['combined_vader_label'] = survey3['combined_vader_compound'].apply(get_vader_label)

# View label counts
print(survey3['review_vader_label'].value_counts())
print(survey3['summary_vader_label'].value_counts())
print(survey3['combined_vader_label'].value_counts())

# #### Visualise the compound scores

fig, axes = plt.subplots(1, 3, figsize=(28, 10))

# Review compound
axes[0].hist(survey3['review_vader_compound'], bins=15,
             facecolor='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].set_title('Review VADER Compound', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Compound Score', fontsize=16,fontweight='bold')
axes[0].set_ylabel('Count', fontsize=16, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[0].axvline(survey3['review_vader_compound'].mean(), 
                color='#E74C3C', linestyle='-', linewidth=1.5, label='Mean')
axes[0].legend()

# Summary compound
axes[1].hist(survey3['summary_vader_compound'], bins=15,
             facecolor='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].set_title('Summary VADER Compound', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Compound Score', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=16, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[1].axvline(survey3['summary_vader_compound'].mean(), 
                color='#3498DB', linestyle='-', linewidth=1.5, label='Mean')
axes[1].legend()

# Combined compound
axes[2].hist(survey3['combined_vader_compound'], bins=15,
             facecolor='#2ECC71', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[2].set_title('Combined VADER Compound', fontsize=16, fontweight='bold')
axes[2].set_xlabel('Compound Score', fontsize=16, fontweight='bold')
axes[2].set_ylabel('Count', fontsize=16, fontweight='bold')
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Neutral')
axes[2].axvline(survey3['combined_vader_compound'].mean(), 
                color='#2ECC71', linestyle='-', linewidth=1.5, label='Mean')
axes[2].legend()

plt.suptitle('VADER Sentiment Compound Scores', fontsize=24, fontweight='bold')
plt.tight_layout()
plt.savefig('vader_sentiment_analysis.png', dpi=300)
plt.show()

# #### Visualise sentiment label distribution

fig, axes = plt.subplots(1, 3, figsize=(28, 10))

colors = ['#2ECC71', '#95A5A6', '#E74C3C']  # positive, neutral, negative

for ax, col, title in zip(axes, 
                           ['review_vader_label', 'summary_vader_label', 'combined_vader_label'],
                           ['Review', 'Summary', 'Combined']):
    counts = survey3[col].value_counts()
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_title(f'{title} Sentiment Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sentiment', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    # Add count labels on top of bars
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(bar.get_height())), ha='center', fontsize=12, fontweight='bold')

plt.suptitle('VADER Sentiment Label Distribution', fontsize=24, fontweight='bold')
plt.tight_layout()
plt.savefig('vader_sentiment_label_distribution.png', dpi=300)
plt.show()

# Top 20 most negative reviews
negative_vader = survey3.nsmallest(20, 'combined_vader_compound')[[
     'summary','review', 
     'combined_vader_compound',
     'combined_vader_label']]

# View with wider columns
negative_vader.style.set_properties(
    subset=['review','summary'], **{'width': '1200px'})

# Top 20 most positive reviews
positive_vader = survey3.nlargest(20, 'combined_vader_compound')[
    ['summary', 'review',
     'combined_vader_compound',
     'combined_vader_label']]

# View with wider columns
positive_vader.style.set_properties(
    subset=['review', 'summary'], **{'width': '1200px'})

# #### Insights from Vader Sentiment Analysis
# - The results of 20 negative and 20 positive reviews using VaderSentiment and TextBlob are slightly different, however the overall theme from findings are quite similar.
# - Top 20 most positive reviews: 1741 reviews: 88.8% of reviews are considered positive: customers are generally quite satisfied with their purchases
# + great value for money: price satisfaction
# + gift purchases:  major purchase purpose
# + fun and enjoyment: entertainment value is high
# + age inclusivity
# + educational/therapeutic: "great for playing great for creative workshops"
# - Top 20 most negative reviews: 129 reviews: 6.6%
# + Product quality: ""crappy cardboard", "bad quality all made of paper", "defective product": Material quality is top complaint
# + Disappointment: "a disappointing coop game"
# + Poor value: "money trap", "not worth the money"
# + Not interested enough
# -> VADER captured better those reviews with stronger negativity and short summaries
# + Review 872 scored -0.905 — the most negative in the dataset — contained multiple strong negative words ("crappy", "disgusting", "shame") which VADER detected very accurately
# + Some cases where misclassification might appear such as review 1766 "ball of whacks very useful for pain management" - might suggest a therapeutic product but flagged as "negative"
# - Findings:
# + Value for money is the biggest positive driver — appearing consistently across top positive reviews.
# + Quality and disappointment are the biggest negative drivers - particularly material quality and unmet expectations.
