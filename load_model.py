import joblib
from sklearn.compose import _column_transformer

# Attempt to load the pipeline
model = joblib.load('order_cancel_pipeline.pkl', mmap_mode=None)

print("Model loaded successfully!")
