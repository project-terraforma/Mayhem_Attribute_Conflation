# Machine Learning Models

This directory contains the trained ML models used for attribute selection.

## Structure
*   `name/`, `phone/`, `website/`, `address/`, `category/`: Subdirectories for each attribute.
*   `best_model_*.joblib`: The serialized model artifact (Gradient Boosting, Random Forest, or Logistic Regression) selected as the best performer.
*   `scaler_*.joblib`: The scaler used to normalize features before inference.
*   `training_summary.json`: Metrics (F1, Accuracy, Time, Memory) for all models trained for that attribute.

## Training
Models are trained using `scripts/train_models.py` on the Synthetic Golden Dataset.
