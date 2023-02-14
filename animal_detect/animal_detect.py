from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet')
model.summary()

#---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)
show_image("happy_dog.jpg")

#---------------------------------------------------------------------------

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_and_process_image(image_path):
    print('Original image shape: ',mpimg.imread(image_path).shape)
    image_s = image_utils.load_img(image_path, target_size=(224,224))
    image_s_array = image_utils.img_to_array(image_s)
    print('image_s_array:',image_s_array.shape)
    image_s_array_reshape = image_s_array.reshape(1,224,224,3)
    image_forVGG16 = preprocess_input(image_s_array_reshape)
    print('Processed image shape: ', image_forVGG16.shape)
    return image_forVGG16

processed_image = load_and_process_image("happy_dog.jpg")

#---------------------------------------------------------------------------

from tensorflow.keras.applications.vgg16 import decode_predictions

def readable_prediction(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    predictions = model.predict(image)
    print('Predicted: ', decode_predictions(predictions, top=3))
readable_prediction("happy_dog.jpg")

readable_prediction("brown_bear.jpg")

readable_prediction("sleepy_cat.jpg")

import numpy as np
def pet_gate(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    predictions = model.predict(image)
    print('predictions: ',predictions)
    top1prediction = np.argmax(predictions)
    print("This prediction:", top1prediction)
    if 151 <= top1prediction <= 268:
        print("Dog. Come in!")
    elif 281 <= top1prediction <= 285:
        print("Cat. Come in!")
    else:
        print("Go away!")

pet_gate("happy_dog.jpg")

pet_gate("brown_bear.jpg")

pet_gate("sleepy_cat.jpg")














