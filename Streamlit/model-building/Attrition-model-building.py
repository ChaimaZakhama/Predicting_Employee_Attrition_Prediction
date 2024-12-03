import pandas as pd #type: ignore
employee_data = pd.read_csv('employee_data_oversampled.csv')

df = employee_data.copy()
target = 'Attrition'
encode = ['JobRole']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'No':0,'Yes':1}
def target_encode(val):
    return target_mapper[val]

df['Attrition'] = df['Attrition'].apply(target_encode)

# Separating X and y
X = df.drop('Attrition', axis=1)
Y = df['Attrition']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier #type: ignore
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('attrition_clf.pkl', 'wb'))
