
# Convolutional Neural Network for Flower Recognition

## Dataset
The dataset contains 4242 images of flowers, 
the pictures are divided into five classes:
- chamomile
- tulip
- rose
- sunflower
- dandelion.

## Setup

### Libraries
```
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import random
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
```

### Configuration
```
IMAGE_SIZE = 250
CHANNELS = 3
BATCH_SIZE = 256
EPOCHS = 20
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
num_training = 3019

TRAIN_PATH = "D:/flowers_recognition/Dataset/splits/train"
VAL_PATH = "D:/flowers_recognition/Dataset/splits/val"
TEST_PATH = "D:/flowers_recognition/Dataset/splits/test"

```

## Data Augmentation
```
train = ImageDataGenerator(rescale = 1./255,
                           rotation_range = 25,
                           shear_range = 0.5,
                           zoom_range = 0.5,
                           width_shift_range = 0.2,
                           height_shift_range=0.2,
                           horizontal_flip=True)

validation = ImageDataGenerator(rescale = 1./255,
                           rotation_range = 25,
                           shear_range = 0.5,
                           zoom_range = 0.5,
                           width_shift_range = 0.2,
                           height_shift_range=0.2,
                           horizontal_flip=True)
                          
```


## Importing Data
```
train_batches = train.flow_from_directory(TRAIN_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="categorical",
                                       seed=42)

val_batches = validation.flow_from_directory(VAL_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="categorical",
                                       seed=42)
```

## Feature Extraction (InceptionV3)
```
inception = InceptionV3(weights="imagenet",
                        include_top=False,
                        input_shape=IMG_SHAPE)

inception.trainable = False
```

## Final Model
```
model = Sequential([
    inception,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(1024, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
```

## Training
```
history = model.fit(train_batches,
                    steps_per_epoch = num_training//BATCH_SIZE,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = val_batches)
```
**Training Accuracy:** 87.3%

## Loss
[![Screenshot-2022-11-12-045510.png](https://i.postimg.cc/QdqzWKCF/Screenshot-2022-11-12-045510.png)](https://postimg.cc/bZsg7JQP)

## Testing
### Data
```
testing = ImageDataGenerator(rescale = 1./255)

test_batches = testing.flow_from_directory(TEST_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="categorical",
                                       seed=42)
```
### Results
```
results = model.evaluate(test_batches)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
```

```
2/2 [==============================] - 6s 5s/step - loss: 0.3177 - accuracy: 0.8813
Test loss: 0.31772616505622864
Test accuracy: 0.8812785148620605
```
## Note
A similar model is deployed in a streamlit app in this [repo](https://github.com/96ibman/streamlit-flower-recognition)

## Authors
- [Ibrahim Nasser](https://github.com/96ibman)


## Connect With Me!
- [Website](https://96ibman.github.io/ibrahim-nasser/)
- [LinkedIn](https://www.linkedin.com/in/ibrahimnasser96/)
- [Twitter](https://twitter.com/mleng_ibrahim)
