# hoplite2
[![PyPI version](https://badge.fury.io/py/hoplite2.svg)](https://badge.fury.io/py/hoplite2)
> a sparsity analysis tool for neural networks

## Installation
`pip install hoplite2`


## Usage
There are 2 main classes that are useful in Hoplite2: Spartan and Hoplite.

The Hoplite Class is the main way to use Hoplite2.
```Python
from hoplite2 import Hoplite
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

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

hop = Hoplite(model, vgg16_preprocess, layers=[
    "block1_conv2"
    "block2_conv2"
    "block3_conv3"
    "block4_conv3"
    "block5_conv3"
])

hop.analyze_file("test.png") # analyzes test.png

hop.output("output.csv") # saves output to file
```


Spartan implements several useful functions to analyze sparsity of arrays.
These functions include:
```
 - compute_average_sparsity(output,equals_zero=lambda x: x == 0)
 - consec_1d(arr, hist, equals_zero=lambda x: x == 0)
 - consec_row(output, equals_zero=lambda x: x == 0)
 - consec_col(output, equals_zero=lambda x: x == 0)
 - consec_chan(output, equals_zero=lambda x: x == 0)
 - vec_1d(arr, vec_size, hist, equals_zero=lambda x: x == 0):
 - vec_3d_row(output, vec_size, equals_zero=lambda x: x == 0):
 - vec_3d_col(output, vec_size, equals_zero=lambda x: x == 0):
 - vec_3d_chan(output, vec_size, equals_zero=lambda x: x == 0):
```

equals_zero is a function that takes in a number and returns True if the number is considered zero. This is useful if you wish to look for values that are also close to zero and allows for additional customizablity.
