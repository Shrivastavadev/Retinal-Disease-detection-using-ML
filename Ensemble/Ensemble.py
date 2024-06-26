# -*- coding: utf-8 -*-
"""Ensemble.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mhD9JqeT2MOKwPovAd2mCzoJgnzwvCt9
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, ResNet152
from tensorflow.keras.layers import Input, Average, Lambda, Dense
from tensorflow.keras.models import Model

# mounting data from G-Drive
from google.colab import drive
drive.mount('/content/drive')
# Paths to the saved model weights
densenet_weights_path = '/content/drive/MyDrive/dataset_ML/RFMiD/history/DensNet121.h5'
resnet_weights_path = '/content/drive/MyDrive/dataset_ML/RFMiD/history/ResNet152.h5'

# Load the trained DenseNet121 model with the specific weights
densenet_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
densenet_model.load_weights(densenet_weights_path)

# Load the trained ResNet152 model with the specific weights
resnet_model = ResNet152(weights=None, include_top=False, input_shape=(224, 224, 3))
resnet_model.load_weights(resnet_weights_path)

# Ensure the layers are not trainable
for layer in densenet_model.layers:
    layer.trainable = False

for layer in resnet_model.layers:
    layer.trainable = False

# Input layer
input_layer = Input(shape=(224, 224, 3))

# Get outputs from each model
densenet_output = densenet_model(input_layer)
resnet_output = resnet_model(input_layer)

# Global Average Pooling to reduce dimensions before averaging
densenet_output = tf.keras.layers.GlobalAveragePooling2D()(densenet_output)
resnet_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)

# Weighted average of the outputs
weighted_densenet_output = Lambda(lambda x: x * 0.6)(densenet_output)
weighted_resnet_output = Lambda(lambda x: x * 0.4)(resnet_output)
weighted_average_output = Average()([weighted_densenet_output, weighted_resnet_output])

# Final dense layer for classification
output_layer = Dense(39, activation='tanh')(weighted_average_output)

# Create ensemble model
ensemble_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print summary of the ensemble model
ensemble_model.summary()

try:
  data =np.load("/content/drive/MyDrive/dataset_ML/RFMiD/Training_Set/Training_Set/temporary/Ensemble_data.npy")
  labels = np.load("/content/drive/MyDrive/dataset_ML/RFMiD/Training_Set/Training_Set/temporary/Ensemble_labels.npy")
except FileNotFoundError:
  data = []
  labels = []

  # Load the CSV file
  df = pd.read_csv(path_RFMiD_CSV)
  # Loop through each row in the CSV file
  for index, row in df.iterrows():
      # Load the image and resize it
      path_ = os.path.join(path_image_train+ "/" + str(row['ID'])+".png")
      print(row["ID"],end=" ")
      image = cv2.imread(path_)
      image = cv2.resize(image, (128, 128))
      print("image data added")
      image = img_to_array(image)
      data.append(image)

      # Get the labels for this image
      labels_row = [row[label] for label in df.columns if label != 'ID']
      label_array = np.array(labels_row)
      labels.append(label_array)

  # Convert the data and labels to numpy arrays
  data = np.array(data, dtype="float32") / 255.0 ##### this step is important as it converts the images pixle value from range 1-0 ######
  labels = np.array(labels)
  path_save_data="/content/drive/MyDrive/dataset_ML/RFMiD/Training_Set/Training_Set/temporary/Ensemble_data.npy"
  path_save_label="/content/drive/MyDrive/dataset_ML/RFMiD/Training_Set/Training_Set/temporary/Ensemble_labels.npy"
  np.save(path_save_data, data)
  np.save(path_save_label, labels)
else:
  print("data and labels loaded")

print(data.shape)
print(labels.shape)

anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

datagen = ImageDataGenerator()


datagen.fit(xtrain)
# Fits-the-model
history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=128),
               steps_per_epoch=xtrain.shape[0] //128,
               epochs=50,
               verbose=2,
               callbacks=[anne, checkpoint],
               validation_data=(xtrain, ytrain))

model_history_path = "/content/drive/MyDrive/dataset_ML/RFMiD/history/Ensemble.h5"
with h5py.File(model_history_path, 'w') as file:
    for key, value in history.history.items():
        file.create_dataset(key, data=value)

ypred = model.predict(xtest)

total = 0
accurate = 0
accurateindex = []
wrongindex = []

for i in range(len(ypred)):
    if np.argmax(ypred[i]) == np.argmax(ytest[i]):
        accurate += 1
        accurateindex.append(i)
    else:
        wrongindex.append(i)

    total += 1

print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')

