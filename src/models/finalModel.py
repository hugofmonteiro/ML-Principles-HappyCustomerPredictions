import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# Load the data
df = pd.read_csv('../../data/ACME-HappinessSurvey2020.csv')

# Create new features based on averages
var_pairs = [
    ('X15', ['X1', 'X5'], 'X152'),
    ('X36', ['X3', 'X5'], 'X362'),
    ('X46', ['X4', 'X6'], 'X462')
]

for avg_var, vars_to_avg, new_var in var_pairs:
    df[avg_var] = df[vars_to_avg].mean(axis=1)
    df[new_var] = (df[avg_var] > 3).astype(int)

# Creating the interaction between variables X2 and the newly created X15
df['Interaction'] = df.iloc[:, 2] * df.iloc[:, 7]
    
# Create features and label
features = df.iloc[:, [2, 7, 8, 9, 10, 12, 13]].values
label = df.iloc[:, 0].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    label, 
                                                    test_size=0.2, 
                                                    random_state=0)

# Random Forest Model
model = RandomForestClassifier(max_depth=None, 
                               min_samples_leaf=1, 
                               min_samples_split=2, 
                               n_estimators=100, 
                               random_state=0)

model.fit(X_train, y_train)

# F1 Score
print("Training Score is {} and Testing Score is {}".format(
    f1_score(y_train, model.predict(X_train)),
    f1_score(y_test, model.predict(X_test))
))

# Feature importances
importances = model.feature_importances_
print("Feature importances:", importances)
