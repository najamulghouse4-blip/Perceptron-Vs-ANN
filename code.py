```notebook-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.linear_model import Perceptron

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target
df['Species'] = df['Species'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
df['Id'] = df.index + 1

X = df.drop(columns = ["Species","Id"])
Y = df["Species"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(Y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y_encoded,test_size = 0.2,stratify = y_encoded,random_state = 42)

per = Perceptron(random_state = 42, max_iter = 1000, )
per.fit(X_train,y_train)

y_predic_per = per.predict(X_test)

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

ann_model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(y_train_encoded.shape[1], activation='softmax')
])

ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

ann_model.fit(X_train, y_train_encoded, epochs=50, batch_size=5, verbose=0)

loss, ann_accuracy = ann_model.evaluate(X_test, y_test_encoded, verbose=0)

print(f"Perceptron Accuracy: {accuracy_score(y_test, y_predic_per):.4f}")
print(f"ANN Accuracy: {ann_accuracy:.4f}")

print("\nPerceptron Classification Report:")
print(classification_report(y_test, y_predic_per))

ann_predictions = ann_model.predict(X_test)
ann_predicted_classes = np.argmax(ann_predictions, axis=1)
print("\nANN Classification Report:")
print(classification_report(y_test, ann_predicted_classes))

plt.style.use('seaborn-v0_8-darkgrid')

accuracy_data = pd.DataFrame({
    'Model': ['Perceptron', 'ANN'],
    'Accuracy': [accuracy_score(y_test, y_predic_per), ann_accuracy]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_data, palette='coolwarm')
plt.ylim(0, 1.05)
plt.title('Perceptron vs. ANN: Accuracy Comparison', fontsize=16)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xlabel('Model', fontsize=12)

for index, row in accuracy_data.iterrows():
    plt.text(index, row.Accuracy + 0.02, f'{row.Accuracy:.4f}', color='black', ha="center", fontsize=11)

plt.tight_layout()
plt.show()
```

i want you to make a streamlit ui for this code and it should be cool,great,eye catching ,jaw dropping
