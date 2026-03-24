import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

data = []
labels = []

dataset_path = "Dataset"
categories = []

# Walk through dataset
for main_cat in os.listdir(dataset_path):
    main_path = os.path.join(dataset_path, main_cat)
    if os.path.isdir(main_path):
        for subcat in os.listdir(main_path):
            sub_path = os.path.join(main_path, subcat)
            if os.path.isdir(sub_path):
                categories.append(f"{main_cat}_{subcat}")
                for img_name in os.listdir(sub_path):
                    img_path = os.path.join(sub_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        data.append(img.flatten())
                        labels.append(f"{main_cat}_{subcat}")

data = np.array(data)
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# Save model
pickle.dump(model, open("sign_model.pkl", "wb"))
print("✅ Model saved as sign_model.pkl")
