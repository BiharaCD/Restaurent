# train.py
import pandas as pd
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

CSV_PATH = 'restaurant_orders.csv'  # put the dataset into project folder

NUMERIC = ['order_amount','number_of_items','distance_km','customer_rating','previous_cancellations','delivery_duration_min']
CATEGORICAL = ['restaurant','city','delivery_type','order_time','day_of_week','payment_method']

df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates().reset_index(drop=True)

# target mapping
df['order_canceled_bin'] = df['order_canceled'].map({'No':0,'Yes':1})

X = df[NUMERIC + CATEGORICAL]
y = df['order_canceled_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
# Explicitly provide both transformers and their target columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, NUMERIC),
        ('cat', cat_pipe, CATEGORICAL)
    ],
    remainder='drop'
)

pipe = Pipeline([('pre', preprocessor), ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))])

print("Training model...")
pipe.fit(X_train, y_train)

print("Evaluating...")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]))

print("Saving model and metadata...")
joblib.dump(pipe, 'order_cancel_pipeline.pkl')

meta = {col: list(df[col].dropna().unique()) for col in CATEGORICAL}
with open('feature_meta.json', 'w') as f:
    json.dump(meta, f)
print("Done.")

