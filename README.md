# X Days of ML Code

[Here](https://github.com/beatobongco/x-days-of-ml-code/blob/master/rules.md) are the rules. This is a modified version of the [100 Days of ML Code challenge](https://github.com/llSourcell/100_Days_of_ML_Code).

### Streak: 2 days

## 2018/9/10
Tags: setup runtime gpu

Colab: https://colab.research.google.com/drive/1y14mcC3RFRY_kL2n9LAZ3tRJ6LhZiR0p

* Engineering tips from the master https://medium.com/@francois.chollet/notes-to-myself-on-software-engineering-c890f16f4e4d
* You can change colab runtime to use GPU! 
* You can setup colab to point to a runtime on a Google Compute Engine instance! https://research.google.com/colaboratory/local-runtimes.html
* Colab is great but just in case you ever need a dev setup again https://hackernoon.com/launch-a-gpu-backed-google-compute-engine-instance-and-setup-tensorflow-keras-and-jupyter-902369ed5272
* learned about `keras.preprocessing.image.ImageDataGenerator` to apply random transformations to augment one's dataset. It is interesting to note however, that the parameter `zoom_range` can actually be bad if you supply it raw images of the class in the wild, e.g. a cat sitting on a sofa. It might get the sofa instead of the cat and train the model with that being attached to the label `cat`. One thing about image classification, you first need good object detection.

## 2018/9/9
Tags: tensorflow read image reading

Colab: https://colab.research.google.com/drive/15yTXLUfoSfIhiGG3G1B4fZtQsqTg9xBU

Working on creating a pipeline that can train a binary image classifier called XorNotX (inspired by hotdog or not hotdog).
Still figuring out which stack to use moving forward. I started with Tensorflow but am leaning on restarting with Keras for simplicity.

* There's a [Tensorflow quirk](https://github.com/tensorflow/tensorflow/issues/1763) where if you use certain image processing functions like `tf.image.resize_images` then `tf.convert_image_dtype` it doesn't scale the range properly. You have to divide the output by 255. You can confirm the output by plotting the image via `matplotlib`
* Tensorflow hub is great! You can pull pretrained models with just one line. It can also be integrated into Keras via `keras.layers.Lambda`. See https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440
* Mounting Google Drive + colab + insync + `tf.WholeFileReader` = easy quick stack for doing ML experiments!
* `tf.data.Dataset.from_generator` + `requests` can be used to feed in images from URLs

Next steps: 
* My understanding of [Graphs and Sessions](https://www.tensorflow.org/guide/graphs) is bad, I can probably devote a bit of time to learning.
* I should probably go through this: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* I think honestly I would learn better by attacking Kaggle problems with my own stack even though I'm sorely tempted to go through courses like https://www.udacity.com/course/deep-learning--ud730. Whatever's the case, daily hands-on learning would still be the best.

