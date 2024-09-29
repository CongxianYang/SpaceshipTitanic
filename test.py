import pandas as pd
from kaggle.SpaceshipTitanic.processing_data import preprocessing
import pickle

with open('models\\randomforestclassifier.pkl', 'rb') as f:
  reloaded_model = pickle.load(f)
# Read in the Test CSV Dataset
test_df = pd.read_csv('Data/test.csv')
# Deep copy
# Run through the preocessing pipeline
preprocessing(test_df)
abt_test = test_df.copy()
# One hot encode categorical variables
abt_test = pd.get_dummies(abt_test.drop('PassengerId', axis=1))

yhat_test = reloaded_model.predict(abt_test)


submission = pd.DataFrame([test_df['PassengerId'], yhat_test]).T
submission.columns = ['PassengerID', 'Transported']
submission.to_csv('submission\kaggle_submission.csv', index=False)