import logging
import sys
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# FIXME TODO: update number of epochs here
# FYI: each epoch takes around 8k seconds in a CRC single GPU
EPOCHS = 2


def setup_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(module)s:%(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


logger = setup_logger()
logger.info("Starting training of VGG19 with imagenette dataset.")

# print GPU devices configuration
gpu_devices = tf.config.list_physical_devices('GPU')
logger.info(f"Retrieved GPU devices: {gpu_devices}")
if gpu_devices:
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    logger.info(f"GPU details: {details}")

# doc: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
tfds.disable_progress_bar()
(ds_train, ds_val), ds_info = tfds.load(
    'imagenette/160px-v2',  # imagenette/160px-v2
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

# verbose 1=progress bar; 2=one line per epoch
history = model.fit(ds_train, epochs=EPOCHS, verbose=2, validation_data=ds_val,
                    workers=4, use_multiprocessing=True)

logger.info(f"Finished training with output metrics: {model.get_metrics_result()}")

for layer in model.layers[:11]: # You can choose a different layer number for fine-tuning
    layer.trainable = False
for layer in model.layers[11:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine_tune = model.fit(ds_train, epochs=EPOCHS, verbose=2, validation_data=ds_val,
                              workers=4, use_multiprocessing=True)

logger.info("Finished training of VGG19 with imagenette dataset.")
logger.info(f"Model summary: {model.summary()}")
# saving and loading this model: https://www.tensorflow.org/api_docs/python/tf/saved_model/save
# TODO: you need to create a directory in your HOME dir, named "model_vgg"
# NOTE: relative path seems not to work
model.save("vgg19_trained", "/afs/crc.nd.edu/user/a/amaltar2/model_vgg")
