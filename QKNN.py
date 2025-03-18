import pennylane as qml
from pennylane import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

n_qubits = 4
# Generate a simple binary dataset
df = pd.read_csv("/home/ainazj1/.cache/kagglehub/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset/versions/1/processed_data.csv")
df = df.dropna(subset=['message'])

labels = df["label"]
X = df["message"]

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
    
vectorizer = TfidfVectorizer(max_features=1000)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Apply PCA to reduce dimensions for training data and transform validation data using the same PCA
pca = PCA(n_components=n_qubits)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Quantum feature embedding (Quantum Data Processing)

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_embedding(x):
    """Quantum feature encoding circuit"""
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
    qml.BasicEntanglerLayers(weights=np.random.randn(3, n_qubits), wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Apply quantum transformation to the dataset
X_train_q = np.array([quantum_embedding(xi) for xi in X_train])
X_test_q = np.array([quantum_embedding(xi) for xi in X_test])

# Train a classical classifier (SVM)
clf = SVC(kernel="linear")
clf.fit(X_train_q, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_q)
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"Quantum Processed + Classical SVM Accuracy: {accuracy:.2f}")

