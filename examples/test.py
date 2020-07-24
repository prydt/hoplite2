#!/usr/bin/env python3
from hoplite2 import Hoplite
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# keras model to analyze
model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# preprocess function
def vgg16_preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    return preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))


hop = Hoplite(
    model,
    vgg16_preprocess,
    layers=[
        "block1_conv2",
        "block2_conv2",
        "block3_conv3",
        "block4_conv3",
        "block5_conv3",
    ],
)

hop.analyze_file("../test2/dog.jpg")  # analyzes test.png

hop.output("output.csv")  # saves output to file
