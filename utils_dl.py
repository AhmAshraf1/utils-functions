import time
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import random
import os
import pathlib
import PIL.Image
import numpy as np
import seaborn as sns
import plotly.express as px
# from tensorflow import keras
# import tensorflow.keras.utils.image_dataset_from_directory
# import tensorflow.keras.preprocessing.image.ImageDataGenerator


# Function to calculate Inference time for a model to predict
def measure_inference_time(model, input_data):
    """
  Measures the inference time for a TensorFlow model.

  Args:
      model: The TensorFlow model to measure.
      input_data: A batch of input data for the model.

  Returns:
      The average inference time in milliseconds.
    """
    start_time = time.time()
    for _ in range(10):  # Run the inference multiple times for averaging
        model.predict(input_data)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) * 1000 / 10  # Convert to milliseconds and average

    return avg_inference_time

# inference_time = measure_inference_time(model, image)
# print(f"Average inference time: {inference_time:.2f} ms")


# Function to plot the loss and accuracy of the model through epochs
def plot_loss_accuracy(history):
    """Plots the loss and accuracy of training and validation sets.

    Args:
        history: A dictionary containing the model's training history,
            typically returned by model.fit() in TensorFlow or Keras.

    Returns:
        None
    """

    # Ensure presence of required keys for robustness
    required_keys = {'loss', 'val_loss', 'accuracy', 'val_accuracy'}
    if not required_keys.issubset(history.history.keys()):
        raise ValueError("Invalid history object. Missing keys: {}".format(required_keys - history.history.keys()))

    df_loss_acc = pd.DataFrame(history.history)
    df_loss = df_loss_acc[['loss', 'val_loss']]
    df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)

    df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
    df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    df_loss.plot(ax=axes[0], title='Model Loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss')

    df_acc.plot(ax=axes[1], title='Model Accuracy')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy')

    plt.tight_layout()
    plt.show()


# Function to plot the loss and accuracy of the model through epochs
def plot_history(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = history.epoch

    acc = history.history['accuracy']
    val_acc =history.history['val_accuracy']

    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training_loss vs Validation_loss')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training_accuracy vs Validation_accuracy')


# Function to split the dataset
def train_val_test_data(batch_size, img_size, train_directory, validation_directory, test_directory):

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_directory,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=img_size,
                                                                seed=42)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_directory,
                                                                     shuffle=True,
                                                                     batch_size=batch_size,
                                                                     image_size=img_size,
                                                                     seed=42)

    test_dataset = tf.keras.utils.image_dataset_from_directory(test_directory,
                                                               batch_size=batch_size,
                                                               image_size=img_size,
                                                               seed=42)

    class_names = train_dataset.class_names
    return train_dataset, validation_dataset, test_dataset, class_names


# function used to visualize train, validation, test data
def visualize_data(data):
    class_names = data.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].np.astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


# Function to load image using URL and plot with predicted class
def load_prep(img_path, img_title):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image)
    image = tf.image.resize(image,size=(224,224))

    plt.imshow(image / 255.)
    plt.title(img_title)
    plt.suptitle(image.shape)


################################################################

# class_names = test_dataset.class_names
# pred = feature_model.predict(tf.expand_dims(image,axis=0))
# pred
# predicted_value = class_names[pred.argmax()]
# predicted_value

###############################################################
# Function to load random images using Dataset Kaggle URL and plot with predicted class
def random_image_predict(model, test_dir=test_directory, data, rand_class=True, cls_name=None):
    class_names = data.class_names
    if rand_class == True:
        ran_cls = random.randint(0,len(class_names))
        cls = class_names[ran_cls]
        ran_path = test_dir +'/'+ cls+ '/'+ random.choice(os.listdir(test_directory+'/'+cls))
    else:
        cls = class_names[cls_name]
        ran_path = test_dir +'/'+ cls + '/'+ random.choice(os.listdir(test_directory+'/'+cls))
  
    prep_img = load_prep(ran_path)

    pred = model.predict(tf.expand_dims(prep_img,axis=0))
    pred_cls = class_names[pred[0].argmax()]
    pred_percent = pred[0][pred[0].argmax()]*100
    plt.imshow(prep_img/255.)
    if pred_cls == cls:
        c = 'g'
    else:
        c = 'r'
    plt.title(f'actual:{cls},\npred:{pred_cls},\nprob:{pred_percent:.2f}%',color = c ,fontdict={'fontsize':10})
    plt.axis(False)
    
# random_image_predict(feature_model)

# plt.figure(figsize=(15,15))
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     random_image_predict(feature_model,test_directory, train_dataset)

################################################################

data_dir='Kaggle Images File (Directory) URL'
plt.figure(figsize=(15,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    rn = random.choice(os.listdir(data_dir))
    image_path = os.path.join(data_dir,rn)
    img = load_prep(image_path)
    pred = feature_model.predict(tf.expand_dims(img,axis=0))
    pred_name = class_names[pred.argmax()]
    plt.imshow(img/255.)
    plt.title(f'true:{rn} \npred_class:{pred_name}')
    plt.axis(False)
    
################################################################

def predict_img(img_path,model=feature_model, data):
    class_names = data.class_names

    img = load_prep(img_path)

    pred = model.predict(tf.expand_dims(img,axis=0))

    pred_name = class_names[pred.argmax()]

    plt.imshow(img/255.)
    plt.title(f'predicted_class : {pred_name}' + str(pred))
    plt.axis(False)
    
################################################################
# download the image using URL

# !wget "URL"
# predict_img('Name')

################################################################
# Reads an image from a file, decodes it into a dense tensor, and resizes to a fixed shape.
def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = parts[-2]

  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [128, 128])
  return image, label

file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
show(image, label)
####
images_ds = train_dataset.map(parse_image)
for image, label in images_ds.take(2):
    show(image, label)

################################################################
# Predict on new data
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

################################################################
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

################################################################
# Run the TensorFlow Lite model
# Load the model
TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

# Print the signatures from the converted model to obtain the names of the inputs (and outputs)
interpreter.get_signature_list()

# test the loaded TensorFlow Model by performing inference on a sample image by passing the signature name
classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite

# tensorized that image and saved it as img_array
predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)

# The prediction generated by the lite model should be almost identical to the predictions generated by the original model
print(np.max(np.abs(predictions - predictions_lite)))