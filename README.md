# X Days of ML Code

[Here](https://github.com/beatobongco/x-days-of-ml-code/blob/master/rules.md) are the rules. This is a modified version of the [100 Days of ML Code challenge](https://github.com/llSourcell/100_Days_of_ML_Code).

### Current streak: 11 days
### Best streak: 11 days

## 2018/9/20
Tags: backprop

Listened to Ilya Sutskever https://www.youtube.com/watch?v=9EN_HoEk3KY

* Why do neural networks work? All conceivable regularity can be expressed by a short program
* Backprop solves a profound problem, *circuit search*. These are not perfect but can be used to solve many interesting problems.

## 2018/9/19
Tags: colab runtime gce image adam optimizer

Colab: https://colab.research.google.com/drive/1VeVqxRcTLP550AlE_ceIiSkrxqjJ_HQY

* TIL there is a company that sells desktop and laptops for deep learning https://lambdalabs.com/laptops/tensorbook/customize

**Adam**
https://www.youtube.com/watch?v=JXQT_vxqwIs
* Adaptive moment estimation
* It's rmsprop + momentum -- let's study momentum first.

**Momentum**
* https://www.youtube.com/watch?v=k8fTYJPd3_I
* accelerates SGD in the relevant direction and dampens oscillations 
* when you average out gradients, you'll see that oscillations in the noisy directions average out close to 0

**Next steps**
* Use tensorflow hub mobilenet v2 to go through second part of [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), see if you can get >90% validation accuracy and quick inference time without having to cache the bottleneck features
* Watch Lex Fridman's interviews with Ilya and Ray!
* You can use your own runtime as backend for colab notebooks
  * https://blog.kovalevskyi.com/deep-learning-images-for-google-cloud-engine-the-definitive-guide-bc74f5fb02bc
  * https://blog.kovalevskyi.com/gce-deeplearning-images-as-a-backend-for-google-colaboratory-bc4903d24947

## 2018/9/18
Tags: podcast keras fit generator

Colab: https://colab.research.google.com/drive/1nU_VOPGPu1VBUu1G2rkZZNXQkWY9p8Zs

* `fit_generator` trains the model on data generated batch-by-batch by a Python generator or `Sequence`. `ImageDataGenerators` are perfect food for this function
 * basically use this over `fit` to not keep the whole dataset in memory. All the benefits of `generator`s here!
 * runs in parallel to the model for efficiency, allows real-time data augmentation (again, `ImageDataGenerator`) on images on CPU in parallel to training your model on GPU. Finally, we can make use of lots of CPU's on GCP!
 * `keras.utils.Sequence` guarantees ordering and single use of every input per epoch when using `use_multiprocessing=True` argument
* training with the colab runtime with GPU is super fast after the first epoch! Just ~25s per epoch!

**Next steps**
* Still go through [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* Checked out 2 episodes of the [MIT AI Podcast](https://lexfridman.com/ai/). I would love to watch the actual YouTube lectures and write down what I learn. Very good signals from both.
  * Ilya Sutskever https://www.youtube.com/watch?v=9EN_HoEk3KY
  * Ray Kurzweil https://www.youtube.com/watch?v=9Z06rY3uvGY

## 2018/9/17
Tags: rmsprop optimizer recurrent neural networks RNN

Videos:
* https://www.youtube.com/watch?v=_e-LFe_igno
* https://www.youtube.com/watch?v=defQQqkXEfE

![image](https://user-images.githubusercontent.com/3739702/45638182-02aa7900-badf-11e8-83a7-d2b48c241f6c.png)


* reduces oscillations in gradient descent by dividing the gradient with the exponentially weighted average of its recent magnitude (if you look at the formula, you'll see why it's called ROOT MEAN SQUARE)
* vs vanilla gradient descent, allows us to use larger learning rates (doesn't decay with default params)
* debuted in Geoff Hinton's Neural Networks course on Coursera (would be good to take this)
* according to [keras docs](https://keras.io/optimizers/#rmsprop), it's a good choice for recurrent neural networks
* why did fchollet use rmsprop for the image classification tutorial? Let's try `adam`

**Next steps**
* Still go through [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

## 2018/9/16
Tags: python3 pathlib dataset splitting iterator

Gist: https://gist.github.com/beatobongco/e66dde2568bafb68d25b3712753a09e4

* created a script to copy images from the [cats vs. dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data) into the number folder structure detailed in the keras tutorial
* learned Python's `iter()` can be used on lists to make them iterators, which returns items until exhausted. This is quite useful when you want to utilize each item in the list only once, or to have a list that "remembers" where it is when you loop through it in different parts of the code
* `pathlib.Path` is a super cool Python 3 standard library module where you can 
  * compose paths like `Path.cwd() / 'dataset' / 'dog'`
  * create dirs if they don't exist easily via `Path.mkdir(exist_ok=True)`
  * use `glob` syntax to match files/folders like `Path.cwd().glob('cat*')` to get all matches of cat in the current directory!


**Next steps**
* Still go through [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

## 2018/9/15
Tags: max pooling 

[Video](https://www.youtube.com/watch?v=ZjM_XQa5s6s) was very good. Sliding window of a certain size that gets max and puts into an output matrix, effectively shrinking the matrix

Effect of `MaxPooling2D(pool_size=(2, 2), stride=2)`:

```
[4, 3, 8, 5
 9, 1, 3, 6 ==> [9, 8
 6, 3, 5, 3      6, 5]
 2, 5, 2, 5]
```

It has no trainable params!

**Next steps**
* check out `glob` and `pathlib` to split dataset
* Still go through [keras image classification tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

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

