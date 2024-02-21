import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(f"{base_dir}/input/ChurnPredictionMinusBatch.csv")
    
    df.drop(columns=["RowNumber", "Surname", "CustomerId"], inplace=True)
    df = pd.get_dummies(df)
    df.drop(columns="Gender_Male", inplace=True) # redundant column
    
    # Split features and target
    y = df.pop("Exited")
    X = df
    
    # Split data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    # Save data to CSV files
    train_path = f"{base_dir}/train/train.csv"
    pd.concat([y_train, X_train], axis=1).to_csv(train_path, header=False, index=False)

    validation_path = f"{base_dir}/validation/validation.csv"
    pd.concat([y_val, X_val], axis=1).to_csv(validation_path, header=False, index=False)

    test_path = f"{base_dir}/test/test.csv"
    pd.concat([y_test, X_test], axis=1).to_csv(test_path, header=False, index=False)
