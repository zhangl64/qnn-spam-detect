
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("/home/ainazj1/.cache/kagglehub/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset/versions/1/processed_data.csv")
df = df.dropna(subset=['message'])

labels = df["label"]
X = df["message"]

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Define number of qubits
n_qubits = 8

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_embedding(x):
    """Quantum feature encoding circuit"""
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
    qml.BasicEntanglerLayers(weights=np.random.randn(3, n_qubits), wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Apply 5-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

c = 1
for train_index, test_index in kf.split(X, y):
    print(f"Fold {c} in progress")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Apply vectorization inside the loop to prevent data leakage
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    
    # Apply PCA to reduce dimensions inside the loop
    pca = PCA(n_components=n_qubits)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Apply quantum transformation
    X_train_q = np.array([quantum_embedding(xi) for xi in X_train])
    X_test_q = np.array([quantum_embedding(xi) for xi in X_test])
    
    # Train a classical classifier (SVM)
    clf = SVC(kernel="linear")
    clf.fit(X_train_q, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test_q)
    print(accuracy_score(y_test, y_pred))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average="binary"))
    recall_scores.append(recall_score(y_test, y_pred, average="binary"))
    f1_scores.append(f1_score(y_test, y_pred, average="binary"))

# Calculate and display average metrics
avg_metrics = {
    "Accuracy": np.mean(accuracy_scores),
    "Precision": np.mean(precision_scores),
    "Recall": np.mean(recall_scores),
    "F1 Score": np.mean(f1_scores),
}

print(avg_metrics)
