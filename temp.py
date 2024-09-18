import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NaiveBayesClassifier:
    def __init__(self, laplace_smoothing=1e-5):
        self.class_probs = {}
        self.feature_probs = {}
        self.laplace_smoothing = laplace_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Calculate class probabilities
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / n_samples

        # Calculate feature probabilities for each class
        for c in self.classes:
            self.feature_probs[c] = {}
            X_c = X[y == c]
            for feature in range(n_features):
                self.feature_probs[c][feature] = {}
                values, counts = np.unique(X_c[:, feature], return_counts=True)
                for value, count in zip(values, counts):
                    self.feature_probs[c][feature][value] = (count + 1) / (len(X_c) + len(np.unique(X[:, feature])))

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {}
            for c in self.classes:
                class_scores[c] = np.log(self.class_probs[c])
                for feature, value in enumerate(sample):
                    if value in self.feature_probs[c][feature]:
                        class_scores[c] += np.log(self.feature_probs[c][feature][value])
                    else:
                        class_scores[c] += np.log(self.laplace_smoothing)  # Laplace smoothing
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

def load_and_preprocess_data(file_path):
    column_names = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
        'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
        'goiter', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'TSH',
        'T3_measured', 'T3', 'TT4_measured', 'TT4', 'T4U_measured', 'T4U',
        'FTI_measured', 'FTI', 'TBG_measured', 'TBG', 'referral_source', 'diagnosis'
    ]
    
    data = pd.read_csv(file_path, names=column_names, na_values='?')
    data = data.fillna(data.mode().iloc[0])  # Use mode to fill missing values for categorical features
    
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])
    
    features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age', 'sex', 'goiter']
    X = data[features]
    y = data['diagnosis']
    
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    return accuracy, precision, recall, f1

def main():
    X, y = load_and_preprocess_data('thyroid0387.data')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = NaiveBayesClassifier()
    model.fit(X_train.values, y_train.values)
    
    y_pred = model.predict(X_test.values)
    
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
    if accuracy >= 0.85:
        print("The model achieved the expected accuracy of 85% or higher.")
    else:
        print("The model did not achieve the expected accuracy. Further optimization may be needed.")
