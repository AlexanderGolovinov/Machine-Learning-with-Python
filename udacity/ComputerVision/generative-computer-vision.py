import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

faces = fetch_lfw_people(min_faces_per_person=10, color=True)

X = faces.images
y = faces.target

IMG_H = X.shape[1]
IMG_W = X.shape[2]
N_IDENTITIES = faces.target_names.shape[0]

#  Split the data into a training and testing set, with 20% of the data for testing. Use a random_state of 42.
# Hint: use train_test_split from sklearn.model_selection (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'{X_train.shape=}')
print(f'{X_test.shape=}')
print(f'{y_train.shape=}')
print(f'{y_test.shape=}')
print('Number of identities:', N_IDENTITIES)

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Use PCA to reduce the dimensionality of the data and consider only the most important features
pca = PCA(n_components=150)
# Use a linear support vector classifier to classify the faces
svc = LinearSVC()

# Combine the two into a single pipeline for simplicity
model = make_pipeline(pca, svc)

X_train_features = X_train.reshape(-1, IMG_H * IMG_W * 3)
X_test_features = X_test.reshape(-1, IMG_H * IMG_W * 3)

# Fit the model to the training data
model.fit(X_train_features, y_train)

# Evaluate the model
eval_acc = model.score(X_test_features, y_test)
print('Evaluation accuracy:', eval_acc)

# import face_recognition

face_locations = [(0, IMG_W, IMG_H, 0)]

#  convert images into the right format (0-255, 8-bit unsigned integers)
imgs_train = (X_train * 255).astype(np.uint8)
imgs_test = (X_test * 255).astype(np.uint8)

train_embs = np.zeros((len(imgs_train), 128))
for i, img in enumerate(imgs_train):
    #  compute the embeddings for the training images
    embs = face_recognition.face_encodings(img, face_locations)
    train_embs[i] = embs[0]

#  create and train a linear support vector classifier (LinearSVC) on the embeddings (train_embs) and the labels (y_train)
a = LinearSVC()
a.fit(train_embs, y_train)

#  compute the accuracy on the test set. Make sure to featurize the test images first, the same as the training images
test_embs = np.zeros((len(imgs_test), 128))
for i, img in enumerate(imgs_test):
    # compute the embeddings for the testing images
    embs = face_recognition.face_encodings(img, face_locations)
    test_embs[i] = embs[0]

print('Accuracy with DLIB:', a.score(test_embs, y_test))