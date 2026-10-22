import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv('titanic.csv')

df['SexCode'] = (df['Sex'] == 'male').astype('int')
df['FamilySize'] = df['SibSp'] + df['Parch']
df['HasCabin'] = df['Cabin'].notna().astype(int)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 80], labels=False)
df['Embarked'] = df['Embarked'].replace('n', 'S')
df['EmbarkedCode'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['IsFemale1st'] = ((df['Sex'] == 'female') & (df['Pclass'] == 1)).astype(int)
features = ['Pclass', 'SexCode', 'AgeBin', 'Fare',
            'FamilySize', 'HasCabin', 'EmbarkedCode',
            'IsFemale1st']
y = df['Survived'].values
X = df[features].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model1 = LogisticRegression(C=1e10, max_iter=1000)
model1.fit(X, y)
scores = cross_val_score(model1, X, y, cv=kf, scoring='accuracy')
print('模型1准确率：', round(model1.score(X, y), 4),
      '交叉验证得分：', round(scores.mean(), 4))

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X, y)
scores = cross_val_score(model2, X, y, cv=kf, scoring='accuracy')
print('模型2准确率：', round(model2.score(X, y), 4),
      '交叉验证得分：', round(scores.mean(), 4))

model3 = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=5)
model3.fit(X, y)
scores = cross_val_score(model3, X, y, cv=kf, scoring='accuracy')
print('模型3准确率：', round(model3.score(X, y), 4),
      '交叉验证得分：', round(scores.mean(), 4))

model4 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
                                        max_depth=3, random_state=42)
model4.fit(X, y)
scores = cross_val_score(model4, X, y, cv=kf, scoring='accuracy')
print('模型4准确率：', round(model4.score(X, y), 4),
      '交叉验证得分：', round(scores.mean(), 4))
