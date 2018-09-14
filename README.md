# X Days of ML Code

[Here](https://github.com/beatobongco/x-days-of-ml-code/blob/master/rules.md) are the rules. This is a modified version of the [100 Days of ML Code challenge](https://github.com/llSourcell/100_Days_of_ML_Code).

### Current streak: 6 days
### Best streak: 6 days

## 2018/9/14
Tags: CNN convolution python

**Convolutional layer** ([video](https://www.youtube.com/watch?v=YRhxdVk_sIs))
* contain a number of `filters`, each filter detects a pattern        
  * small, randomly-initialized matrix of size *kernel_size* (keras)
  * this small matrix (kernel?) slides (convolves) over the image trying to detect a pattern
  * during each convolution it will compute the dot product of the kernel with the section of the image and store its value in an output matrix. Remember a dot product returns a single value, so the output matrix will have smaller dimensions than the initial matrix
  * The pattern detection can be seen at work in the ff image. Bright output represents what the filter strongly detects ![image](https://user-images.githubusercontent.com/3739702/45540860-171d1600-b840-11e8-9315-7f1ebeb158a9.png)

**Next steps**
* From the deeplizard videos, I can learn from source: fast.ai lectures
* Check out this max pooling video https://www.youtube.com/watch?v=ZjM_XQa5s6s
* check out `glob` and `pathlib` to split dataset

## 2018/9/13
Tags: keras CNN entropic capacity

Colab: https://colab.research.google.com/drive/1y14mcC3RFRY_kL2n9LAZ3tRJ6LhZiR0p

* entropic capacity - how much information your model is allowed to store, having smaller capacity fights overfitting by focusing on the most significant features found in the data, opposed to a high capacity model that can just store many (irrelevant) features
  * you can reduce entropic capacity by reducing layers and size of each layer, and also through regularization 
* adding layers to a keras `Sequential` model is easy, just like adding to a list. Indeed, you can even initialize with a list `Sequential([layer1, layer2, ...])`
* tensorflow uses the image data format of 'channels_last', e.g. for RGB image 150x150 you'd have shape (150, 150, 3)

**Next steps**
* Still go through [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* learn more high-level intution for *convolutional neural networks* and *max pooling* 
* Split cat and dog data into train/test folders with 1k images per class for train and 400 per class for test. Make a new folder in gdrive for this 

## 2018/9/12
Tags: keras preprocessing image ImageDataGenerator RGB normalization

Colab: https://colab.research.google.com/drive/1y14mcC3RFRY_kL2n9LAZ3tRJ6LhZiR0p

Learned a lot about images today.

* `matplotlib.pyplot.imshow` can take in a PIL image
* `keras.preprocessing.image` as `i` for this example
  * If you load a jpeg with `i.load_img` then do `i.img_to_array` then do `imshow` you'll get a weird image. This is because the values are floats and range from 0-255 and according to [`imshow`'s docs](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html), all values should be in the range of 0-1 for floats and 0-255 for integers otherwise they are clipped. 
    * Casting via `np.array.astype(np.uint8)` works, `uint8` because integer array must be of datatype int8 not int32. 
    * Better solution is to just divide the array by 255 to get floats between 0 and 1. This type of normalization is better for ML systems anyway, and it can be plotted for viz! Discussion of other normalization [here](http://forums.fast.ai/t/images-normalization/4058/2)
  * If you load images via `i.ImageDataGenerator.flow`, just use the output of `i.img_to_array` directly. It is scaled automatically. Warning! If you manually scale it first, it will scale it again giving you bad values!
  * Directory structure is important in keras. Split into train/test/validation beforehand.
  
**Next steps**
* Still go through [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

## 2018/9/11
Tags: keras preprocessing image ImageDataGenerator

Colab: https://colab.research.google.com/drive/1y14mcC3RFRY_kL2n9LAZ3tRJ6LhZiR0p

* learned about `keras.preprocessing.image.ImageDataGenerator` to apply random transformations to augment one's dataset. It is interesting to note however, that the parameter `zoom_range` can actually be bad if you supply it raw images of the class in the wild, e.g. a cat sitting on a sofa. It might get the sofa instead of the cat and train the model with that being attached to the label `cat`. One thing about image classification, you first need good object detection.

## 2018/9/10

Tags: setup runtime gpu tensorflow graph session

Colab: https://colab.research.google.com/drive/1WCzAnzMIvo37G5hzoWA9S-JJNKi6Z2aO

* Engineering tips from the master https://medium.com/@francois.chollet/notes-to-myself-on-software-engineering-c890f16f4e4d
* You can change colab runtime to use GPU! 
* You can setup colab to point to a runtime on a Google Compute Engine instance! https://research.google.com/colaboratory/local-runtimes.html
* Colab is great but just in case you ever need a dev setup again https://hackernoon.com/launch-a-gpu-backed-google-compute-engine-instance-and-setup-tensorflow-keras-and-jupyter-902369ed5272

**Graphs and Sessions**
* doing tf in graphs is low level but have some good benefits, however eager execution might be preferred for initial learning. 
* you start coding in tf by building a graph (use as a context manager) `with tf.Graph().as_default():` and then putting ops inside
* Sessions represent a connection between the program and the runtime. You could point it to a runtime in the args `with tf.Session('grpc://example.org:2222'):`. By default, a new `tf.Session` will be bound to---and only able to run operations in---the current default graph. You can use sessions by nested it inside the graph context manager
* `tf.Session.run` is how to run stuff in the graph. Pass one or more (as a list) operations or tensors in as arguments (they are called `fetches`) e.g. `a, b = sess.run([x, y])`
* `fetches` determine what subgraph of the overall graph must be executed to produce the result
* If you define multiple `fetches`, it wont execute graph one time for each fetch
* `run` can take *feeds* `sess.run(y, {x: [1.0, 2.0, 3.0]})` which can feed values typically into `tf.placeholder`s

**Next steps**
* While reading through Graphs and Sessions, the docs recommend to go a level higher with [`tf.estimator.Estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator). I should read a little about it.
* Learn about tf through eager execution https://tf.wiki/
* Could be useful to reread this if going the low-level path https://betterexplained.com/articles/matrix-multiplication/
* Still go through [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

## 2018/9/9
Tags: tensorflow read image reading

Colab: https://colab.research.google.com/drive/15yTXLUfoSfIhiGG3G1B4fZtQsqTg9xBU

Working on creating a pipeline that can train a binary image classifier called XorNotX (inspired by hotdog or not hotdog).
Still figuring out which stack to use moving forward. I started with Tensorflow but am leaning on restarting with Keras for simplicity.

* There's a [Tensorflow quirk](https://github.com/tensorflow/tensorflow/issues/1763) where if you use certain image processing functions like `tf.image.resize_images` then `tf.convert_image_dtype` it doesn't scale the range properly. You have to divide the output by 255. You can confirm the output by plotting the image via `matplotlib`
* Tensorflow hub is great! You can pull pretrained models with just one line. It can also be integrated into Keras via `keras.layers.Lambda`. See https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440
* Mounting Google Drive + colab + insync + `tf.WholeFileReader` = easy quick stack for doing ML experiments!
* `tf.data.Dataset.from_generator` + `requests` can be used to feed in images from URLs

**Next steps** 
* My understanding of [Graphs and Sessions](https://www.tensorflow.org/guide/graphs) is bad, I can probably devote a bit of time to learning.
* I should probably go through this: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* I think honestly I would learn better by attacking Kaggle problems with my own stack even though I'm sorely tempted to go through courses like https://www.udacity.com/course/deep-learning--ud730. Whatever's the case, daily hands-on learning would still be the best.

