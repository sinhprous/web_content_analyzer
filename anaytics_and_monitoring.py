import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report

url = 'https://docs.google.com/spreadsheets/d/1DcAjDy-pxi6gn5aPdfA39fSo-SF4sDEilcqtdPZf0ic/export?format=csv'
df = pd.read_csv(url)

df['time stamp'] = pd.to_datetime(df['time stamp'], format='%b-%d-%Y %H:%M:%S %p UTC').astype('int64') // 10**9
df['value'] = df['value'].replace('[\$,]', '', regex=True).astype(float)


# Find large transactions
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df['is_large_transaction'] = (df['value'] > (Q3 + 2 * IQR))


# Find rapid transactions
df = df.sort_values(by=['from', 'time stamp'], ascending=[True, True])
df['previous_timestamp'] = df.groupby('from')['time stamp'].shift(1)
df['time_since_last'] = (df['time stamp'] - df['previous_timestamp'])
# If there's no previous transaction, set the time difference to -1
df['time_since_last'].fillna(-1, inplace=True)
df['is_rapid_transaction'] = (df['time_since_last'] <= 1.0) & (df['time_since_last'] > 0)


# NOTE: timestamp itself might not contain information to determine fraud, so we use time_since_last
features = ['from', 'to', 'value', 'method called', 'large transaction', 'rapid transaction', 'time_since_last']
X = df[features]
y = df['fraud transaction']

# NOTE: in this example code, we tried to fit all data (no validation)
model = lgb.LGBMClassifier()
model.fit(X, y)

# Predict fraud
df['predicted_fraud'] = model.predict(X)

print(classification_report(y, df['predicted_fraud']))