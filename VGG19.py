import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

(ds_train, ds_val), ds_info = tfds.load(
    'imagenette',
    split=['train', 'validation'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    image = tf.image.resize(image, [224, 224]) # Resize the image to the required input size for VGG19
    image = tf.cast(image, tf.float32) / 255.0 # Normalize the pixel values
    return image, label

ds_train = ds_train.map(preprocess).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(ds_info.features['label'].num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(ds_train, epochs=10, validation_data=ds_val)

for layer in model.layers[:11]: # You can choose a different layer number for fine-tuning
    layer.trainable = False
for layer in model.layers[11:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine_tune = model.fit(ds_train, epochs=10, validation_data=ds_val)
