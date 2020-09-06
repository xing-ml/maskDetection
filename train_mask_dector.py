from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to load dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
args = vars(ap.parse_args())

#hyperParams
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#Grab the list of images
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#if dataset is too large to load into memory, use HDF5 instead.
#loop over the image
for imagePath in imagePaths:

    #extract the label
    label = imagePath.split(os.path.sep)[-2]

    #load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    #update the data and label lists, respectively
    data.append(image)
    labels.append(label)

#convert data and label lists to numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

#preprocessing labels: one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#split the train & test dataset 0.8:0.2
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

#load MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

#construt the head
#Pool => Flatten => Dense(relu) => Dropout => Dense(softmax)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#place the head on top of baseModel
model = Model(inputs=baseModel.input, outputs=headModel)

#freeze all layers in baseModel
for layer in baseModel.layers:
    layer.trainable = False

#compile model
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

#train the head
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch = len(trainX) // BS,
    validation_data = (testX, testY),
    validation_steps = len(testX) // BS,
    epochs=EPOCHS)

#make prediction
print("[INFO] Evaluating network...")
preIdxs = model.predict(testX, batch_size=BS)
print(preIdxs)

#find the index with largest predicted probability
preIdxs = np.argmax(preIdxs, axis=1)

#classification report
print(classification_report(testY.argmax(axis=1), preIdxs, target_names=lb.classes_))

#serialize the model to disk
print("[INFO] Saving mask detector model...")
model.save(args["model"], save_format="h5")

#plot the training loss and accuracy
N=EPOCHS
plt.style("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])