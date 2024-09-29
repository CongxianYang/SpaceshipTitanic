from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from kaggle.SpaceshipTitanic.processing_data import preprocessing
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
from tqdm import tqdm
data = pd.read_csv("D:\mywork\Proeject\cgft-llm\kaggle\SpaceshipTitanic\Data\\train.csv")
df = pd.DataFrame(data)
print(df.info())
preprocessing(df)
next_df = df.copy()

# Create feature columns
# Drop identifier column

X = next_df.drop(['Transported', 'PassengerId'], axis=1)
# One hot encode
X = pd.get_dummies(X)
# Create target columns
y = next_df['Transported']
# Create training and testing partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
pipelines = {
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1234)),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1234))
}
# pipelines 是一个字典，包含两个管道（pipeline）：
# 'rf'：Random Forest 模型，包含 StandardScaler 和 RandomForestClassifier。
# 'gb'：Gradient Boosting 模型，包含 StandardScaler 和 GradientBoostingClassifier。
GradientBoostingClassifier().get_params()
grid = {
    'rf': {
        'randomforestclassifier__n_estimators': [100, 200, 300,400]
    },
    'gb': {
        'gradientboostingclassifier__n_estimators': [100, 200, 300,400]
    }
}
# Create a blank dictionary to hold the models
fit_models = {}
# Loop through all the algos
for algo, pipeline in tqdm(pipelines.items()):
    print(f'Training the {algo} model.')
    # Create new Grid Search CV Cclass
    model = GridSearchCV(pipeline, grid[algo], n_jobs=-1, cv=10)
    # Train the model
    model.fit(X_train, y_train)
    # Store results inside of the dictionary
    fit_models[algo] = model
# Evaluate the performance of the model
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    precision = precision_score(y_test, yhat)
    recall = recall_score(y_test, yhat)
    print(f'Metrics for {algo}: accuracy- {accuracy}, recall- {recall}, precision- {precision}')
with open('models\\gradientboosted.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)
with open('models\\randomforestclassifier.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)