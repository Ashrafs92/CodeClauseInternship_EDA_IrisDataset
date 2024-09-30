import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Assuming 'data' is your DataFrame containing the Iris dataset
# data = pd.read_csv('path_to_your_iris_data.csv')  # Uncomment this line to load your data

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.3)

# Define features and target variable for training and testing
train_X = train[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']]
train_y = train['Species']
test_X = test[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']]
test_y = test['Species']

# Create and train the Decision Tree model
dtmodel = DecisionTreeClassifier()
dtmodel.fit(train_X, train_y)

# Make predictions
dtpredict = dtmodel.predict(test_X)

# Calculate accuracy
dtaccuracy = metrics.accuracy_score(test_y, dtpredict)
print("Decision Tree Model Accuracy is {:.2f}%".format(dtaccuracy * 100))

# Identify mispredictions
test_preddf = test.copy()
test_preddf['Predicted Species'] = dtpredict
wrongpred = test_preddf.loc[test_preddf['Species'] != test_preddf['Predicted Species']]
print("Mispredictions:\n", wrongpred)
