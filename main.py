import  numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import BernoulliRBM

# Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values

# Define CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the CNN model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train, y_train, epochs=5)

# Evaluate the CNN model
cnn_acc = cnn_model.evaluate(X_test, y_test)[1]
print(f"Accuracy of CNN model: {cnn_acc:.4f}")

# Define SVM model
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('svm', SVC())
])

# Flatten the image data for SVM
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Train the SVM model
svm_model.fit(X_train_flatten, y_train)

# Predict with the SVM model
svm_pred = svm_model.predict(X_test_flatten)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"Accuracy of SVM model: {svm_acc:.4f}")

# Define Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train_flatten, y_train)

# Predict with the Random Forest model
rf_pred = rf_model.predict(X_test_flatten)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Accuracy of Random Forest model: {rf_acc:.4f}")

# Define DBN model (using BernoulliRBM for feature extraction)
rbm = BernoulliRBM(n_components=100, n_iter=10, random_state=0, verbose=True)
dbn_model = Pipeline(steps=[('rbm', rbm), ('svm', SVC())])

# Train the DBN model
dbn_model.fit(X_train_flatten, y_train)

# Predict with the DBN model
dbn_pred = dbn_model.predict(X_test_flatten)
dbn_acc = accuracy_score(y_test, dbn_pred)
print(f"Accuracy of DBN model: {dbn_acc:.4f}")