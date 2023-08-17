import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.layers.experimental import preprocessing

# Preprocessing the Training set
train_ds: tf.data.Dataset = image_dataset_from_directory(
    'dataset/training_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary')

# Applying data augmentation using preprocessing layers
train_data_augmentation = tf.keras.Sequential([
    preprocessing.Rescaling(1./255),
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomRotation(0.2)
])

train_ds: tf.data.Dataset = train_ds.map(lambda x, y: (train_data_augmentation(x, training=True), y))

# Preprocessing the Test set
test_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/test_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary')

test_data_augmentation = tf.keras.Sequential([
    preprocessing.Rescaling(1./255)
])

test_ds: tf.data.Dataset = test_ds.map(lambda x, y: (test_data_augmentation(x, training=False), y))

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection (From there we are coding ANN)
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = train_ds, validation_data = test_ds, epochs = 25)

#  Making a single prediction
import numpy as np
from keras.preprocessing import image
def make_a_prediction(filename:str, model:tf.keras.models.Sequential = cnn):
    test_image = image.load_img(f'dataset/single_prediction/{filename}.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    return prediction

for i in range(12):
    prediction = make_a_prediction(f'cat_or_dog ({i+1})')
    print(f"Image {i+1} is a {prediction}")

# Export the model
cnn.save('model.keras')

# Load the model
from keras.models import load_model
model:tf.keras.models.Sequential = load_model('model.keras', compile=False)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Make a prediction
selection = 5
prediction = make_a_prediction(f'cat_or_dog ({selection})', model)
print(f"Image {selection} is a {prediction}")
