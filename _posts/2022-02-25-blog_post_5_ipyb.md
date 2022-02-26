---
layout: post
title: PIC 16B Blog Post 5!
---

Hello dear reader and welcome back to another blog post! Today's task is to teach a machine learning algorithm to distinguish betweeen pictures of dogs and cats. To do so, we will be making use of a Python package called TensorFlow. Let's get started!

Today's post is loosely based off of the following tutorial: [link](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Load Packages and Obtain Data

First let's go ahead and import the libraries we will need for our task today


```python
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import random
```

The following code creates a TensorFlow dataset that we will use for training, validation, and testing. Our dataset consists of labeled images of dogs and cats. As can be seen by the `batch_size` and `shuffle` variables, each time we extract data, we will shuffle our data and randomly get 32 images from each data set. We now have training data, validation data, and testing data.


```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)


class_names = train_dataset.class_names
```

    Found 2000 files belonging to 2 classes.
    Found 1000 files belonging to 2 classes.


Lastly, this next chunk of code allows us to rapidly read in our data. Don't worry too much about this code right now!


```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

### Working with Datasets

First, let's go ahead and visualize the dataset we are working with! We will take small chunks of data (32 images with labels) using the `take` method in `train_dataset.take(1)`. Let's write a function to take a look at some of the images from our dataset down below.


```python
def plot_images():
  plt.figure(figsize=(10,10))
  cats=[]
  dogs=[]

  for images, labels in train_dataset.take(1):
    for i in range(32):
      if class_names[labels[i]] == "cats":
        cats.append(i)
      else:
        dogs.append(i)
    
    for i in range(6):
      if i<3:
        choice = random.choice(cats)
      else:
        choice = random.choice(dogs)
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[choice].numpy().astype("uint8"))
      plt.title(class_names[labels[choice]])
      plt.axis("off")

plot_images()
```


![image-example.png](/images/output_7_0.png)

   


### Check Label Frequencies

How many images do we have in our dataset? We can compute the answer to this using the code chunk below. Our `labels_iterator` allows us to iterate over each image in our entire dataset. As we iterate over our data, we will sum up the number of images of cats and dogs.


```python
labels_iterator = train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()
cats = 0
dogs = 0
total = 0

for i in labels_iterator:
  if i == 0:
    cats += 1
  else:
    dogs += 1

print("number of cats: " + str(cats))
print("number of dogs: " + str(dogs))
```

    number of cats: 1000
    number of dogs: 1000


Because we have equal amounts of dog and cat images, the baseline model randomly guesses cat 50% of the time and dog the other 50% of time. Therefore the baseline model is 50% accurate. We can definitely do better, so let's work to improve our model!

### First Model
Let's go ahead and experiment a bit with TensorFlow's different methods for creating machine learning models. We will initialize a model using `tf.keras.Sequential` and add various labels. 

The `Conv2D` layers make use of convolutional kernels to extract key features from the training images. As the model trains, it becomes more adept at recognizing the most important features of the images that can be used to determine the difference between a dog and a cat. The `MaxPooling2D` layers then summarize and reduce the amount of data obtained by the previous layer. `Conv2D` and `MaxPooling2D` layers are often used together. We also incorporate `Dropout` layers throughout our model to protect agains overfitting. The `Flatten` layer converts the data from 2D to 1D data. Once our data is 1D, we can then pass it through the `Dense` layers which make the actual prediction as to what the image is. The final layer is `layers.Dense(2)` because we have 2 classes of data: dogs and cats.

Before deciding on using this specific model, I did some experimenting to improve the accuracy! I added and removed Conv2D and MaxPooling2D layers as well as changed the parameters in the Dropout layer.


```python
model1 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2),
])
```

Now we need to compile and fit our model to our dataset.



```python
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 7s 92ms/step - loss: 43.8020 - accuracy: 0.5125 - val_loss: 0.6918 - val_accuracy: 0.4839
    Epoch 2/20
    63/63 [==============================] - 5s 83ms/step - loss: 0.6862 - accuracy: 0.5375 - val_loss: 0.6874 - val_accuracy: 0.5495
    Epoch 3/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.6744 - accuracy: 0.5730 - val_loss: 0.6888 - val_accuracy: 0.5223
    Epoch 4/20
    63/63 [==============================] - 7s 102ms/step - loss: 0.6447 - accuracy: 0.5895 - val_loss: 0.7047 - val_accuracy: 0.5186
    Epoch 5/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.5983 - accuracy: 0.6475 - val_loss: 0.7585 - val_accuracy: 0.5260
    Epoch 6/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.5699 - accuracy: 0.6600 - val_loss: 0.8205 - val_accuracy: 0.5235
    Epoch 7/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.5310 - accuracy: 0.6960 - val_loss: 0.8539 - val_accuracy: 0.5149
    Epoch 8/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.4772 - accuracy: 0.7455 - val_loss: 0.9295 - val_accuracy: 0.5309
    Epoch 9/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.4494 - accuracy: 0.7630 - val_loss: 1.2150 - val_accuracy: 0.5136
    Epoch 10/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.4126 - accuracy: 0.7765 - val_loss: 1.1914 - val_accuracy: 0.5309
    Epoch 11/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.3982 - accuracy: 0.8035 - val_loss: 1.4485 - val_accuracy: 0.5272
    Epoch 12/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.3537 - accuracy: 0.8135 - val_loss: 1.4166 - val_accuracy: 0.5248
    Epoch 13/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.3118 - accuracy: 0.8405 - val_loss: 1.6456 - val_accuracy: 0.5260
    Epoch 14/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.2981 - accuracy: 0.8660 - val_loss: 1.6639 - val_accuracy: 0.5396
    Epoch 15/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.2539 - accuracy: 0.8765 - val_loss: 1.9060 - val_accuracy: 0.5446
    Epoch 16/20
    63/63 [==============================] - 5s 83ms/step - loss: 0.2073 - accuracy: 0.9020 - val_loss: 2.4228 - val_accuracy: 0.5347
    Epoch 17/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.1993 - accuracy: 0.9065 - val_loss: 2.3634 - val_accuracy: 0.5384
    Epoch 18/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.1736 - accuracy: 0.9260 - val_loss: 2.3587 - val_accuracy: 0.5099
    Epoch 19/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.1592 - accuracy: 0.9300 - val_loss: 2.7987 - val_accuracy: 0.5309
    Epoch 20/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.1305 - accuracy: 0.9455 - val_loss: 2.9975 - val_accuracy: 0.5483


As can be seen below, **the accuracy of our model stabilized to aaround 54% during training.** This is around 10% better than our baseline model, so we are somewhat pleased by this. However, it looks like our validation accuracy is significantly lower than our accuracy during training. Therefore we suspect that overfitting may be occurring in `model1`.


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f4fa7f08450>




![image-example.png](/images/output_15_1.png)
    
    


Lastly, we will go ahead and evaluate our model on our unseen testing data. As can be seen, we got an accuracy of 55%! This is slightly better than our baseline model, but it is clear that overfitting is a huge issue. 


```python
model1_score = model1.evaluate(test_dataset)
```

    6/6 [==============================] - 1s 51ms/step - loss: 2.4252 - accuracy: 0.5573


### Model with Data Augmentation

Our next step is to add some data augmentation layers to our model. In doing so, we slightly modify images in our training set by rotating or flipping the images. We do this because we need our model to be able to distinguish between dogs and cats regardless of the orientation of the image. By adding the following layers, we will hopefully increase our testing accuracy. 


For our first layer, we will flip our image by reflecting across the y axis. The plot below is a visualization of what theser flipped images look like.


```python
flip = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical")
])
```


```python
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = flip(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```


![image-example.png](/images/output_20_0.png)

    


Our next layer randomly rotates our image. Once again we will visualize what this layer does with our images on the plot below


```python
random_rotation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.5),
])
```


```python
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = random_rotation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
```


    

![image-example.png](/images/output_23_0.png)



Finally, let's incorporate these new layers into a new model! We will follow the same process as we did for model 1: initialize the model, compile the model, visualize the training history, and then finally find the testing accuracy of the model.


```python
model2 = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),                         
    tf.keras.layers.RandomRotation(0.2),                    
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2),
])
```

As can be seen, the **validation accuracy of `model2` during training stabilizes to around 55%.** This validation accuracy is actually slightly lower than the validation accuracy we obtained with `model1`. The validation accuracy is still lower than the accuracy when we train our model, however the difference in values is not quite as significant. Overfitting is of course still a possible concern, but this model is most likely less overfit than `model1` was.


```python
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 7s 89ms/step - loss: 56.3529 - accuracy: 0.4840 - val_loss: 0.6932 - val_accuracy: 0.4901
    Epoch 2/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6933 - val_accuracy: 0.4913
    Epoch 3/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.4901
    Epoch 4/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.4876
    Epoch 5/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.4876
    Epoch 6/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.6937 - accuracy: 0.4705 - val_loss: 0.6932 - val_accuracy: 0.4814
    Epoch 7/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.4810 - val_loss: 0.6931 - val_accuracy: 0.5087
    Epoch 8/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.6932 - accuracy: 0.4880 - val_loss: 0.6931 - val_accuracy: 0.5161
    Epoch 9/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.4860 - val_loss: 0.6931 - val_accuracy: 0.5037
    Epoch 10/20
    63/63 [==============================] - 7s 101ms/step - loss: 0.6932 - accuracy: 0.4850 - val_loss: 0.6931 - val_accuracy: 0.5074
    Epoch 11/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6932 - val_accuracy: 0.4963
    Epoch 12/20
    63/63 [==============================] - 8s 130ms/step - loss: 0.6932 - accuracy: 0.4830 - val_loss: 0.6931 - val_accuracy: 0.4913
    Epoch 13/20
    63/63 [==============================] - 6s 90ms/step - loss: 0.6932 - accuracy: 0.4920 - val_loss: 0.6932 - val_accuracy: 0.4913
    Epoch 14/20
    63/63 [==============================] - 8s 113ms/step - loss: 0.6932 - accuracy: 0.4930 - val_loss: 0.6930 - val_accuracy: 0.5161
    Epoch 15/20
    63/63 [==============================] - 6s 98ms/step - loss: 0.6932 - accuracy: 0.4850 - val_loss: 0.6931 - val_accuracy: 0.5111
    Epoch 16/20
    63/63 [==============================] - 6s 89ms/step - loss: 0.6932 - accuracy: 0.4860 - val_loss: 0.6931 - val_accuracy: 0.5186
    Epoch 17/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.4920 - val_loss: 0.6931 - val_accuracy: 0.5198
    Epoch 18/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.4830 - val_loss: 0.6931 - val_accuracy: 0.5111
    Epoch 19/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6932 - accuracy: 0.4930 - val_loss: 0.6932 - val_accuracy: 0.4901
    Epoch 20/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.6932 - accuracy: 0.4930 - val_loss: 0.6931 - val_accuracy: 0.4938



```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f5026d96950>




    

![image-example.png](/images/output_28_1.png)

    


We also note that the accuracy of the model on our testing data is 52%. This is a bit lower than our accuracy from `model1`, so we can definitely continue to improve upon what we already have.


```python
model2_score = model2.evaluate(test_dataset)
```

    6/6 [==============================] - 1s 56ms/step - loss: 0.6931 - accuracy: 0.5260


### Data Preprocessing

Our next task is to speed up the training speed of our model. Right now the images in our training data have pixels with RGB values between 0 and 255 to distinguish between colors. If we scale these values down to be between 0 and 1, our model will have an easier time training and it will be able to focus more on differentiating between image features. This should increase our accuracy in the long run!


The following code initializes a preprocessing layer which we will add to our model.


```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```


```python
model3 = tf.keras.Sequential([
    preprocessor,
    tf.keras.layers.RandomFlip('horizontal'),                         
    tf.keras.layers.RandomRotation(0.2),                    
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2),
])
```

Now we will train our model and visualize the training history!


```python
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 7s 90ms/step - loss: 0.8495 - accuracy: 0.5055 - val_loss: 0.6905 - val_accuracy: 0.5916
    Epoch 2/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.6807 - accuracy: 0.5490 - val_loss: 0.6430 - val_accuracy: 0.5582
    Epoch 3/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.6573 - accuracy: 0.6085 - val_loss: 0.6485 - val_accuracy: 0.6151
    Epoch 4/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.6323 - accuracy: 0.6350 - val_loss: 0.6424 - val_accuracy: 0.6349
    Epoch 5/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6175 - accuracy: 0.6455 - val_loss: 0.6525 - val_accuracy: 0.6300
    Epoch 6/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.6174 - accuracy: 0.6475 - val_loss: 0.6327 - val_accuracy: 0.6584
    Epoch 7/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.5931 - accuracy: 0.6735 - val_loss: 0.6193 - val_accuracy: 0.6733
    Epoch 8/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.5790 - accuracy: 0.6870 - val_loss: 0.5805 - val_accuracy: 0.6931
    Epoch 9/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.5707 - accuracy: 0.7040 - val_loss: 0.6494 - val_accuracy: 0.6572
    Epoch 10/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.5705 - accuracy: 0.6970 - val_loss: 0.6165 - val_accuracy: 0.6844
    Epoch 11/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5504 - accuracy: 0.7105 - val_loss: 0.5808 - val_accuracy: 0.7203
    Epoch 12/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5465 - accuracy: 0.7135 - val_loss: 0.5840 - val_accuracy: 0.7054
    Epoch 13/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.5439 - accuracy: 0.7175 - val_loss: 0.6236 - val_accuracy: 0.6844
    Epoch 14/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.5382 - accuracy: 0.7300 - val_loss: 0.6342 - val_accuracy: 0.6869
    Epoch 15/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.5199 - accuracy: 0.7320 - val_loss: 0.5616 - val_accuracy: 0.7252
    Epoch 16/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.5385 - accuracy: 0.7245 - val_loss: 0.5526 - val_accuracy: 0.7067
    Epoch 17/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5194 - accuracy: 0.7335 - val_loss: 0.5443 - val_accuracy: 0.7327
    Epoch 18/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.4966 - accuracy: 0.7575 - val_loss: 0.5724 - val_accuracy: 0.7252
    Epoch 19/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.4995 - accuracy: 0.7550 - val_loss: 0.5541 - val_accuracy: 0.7129
    Epoch 20/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.4957 - accuracy: 0.7550 - val_loss: 0.5755 - val_accuracy: 0.7166



```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f4fa7ae0e10>




    

![image-example.png](/images/output_36_1.png)

    


We are pretty pleased with these results, as the **validation accuracy of our model stabilized to around 72%.** This is a significant improvement from our baseline model which only had an accuracy of 50%. Similar to `model2`, it is definitely possible that overfitting is still happening, as the validation accuracy was often lower than the training accuracy when we compiled our model. However, the two values were relatively similar throughout the training process, so we were able to minimize at least some of the overfitting.


```python
model3_score = model3.evaluate(test_dataset)
```

    6/6 [==============================] - 1s 54ms/step - loss: 0.4978 - accuracy: 0.7708


### Transfer Learning

For our last model we will make use of someone else'e machine learning model and apply it to our task. The following code initializes the base_model_layer from MobileNextV2. We will then incorporate the layer into a model specific to our task and follow the same process that we did for the past 3 models!


```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```


```python
model4 = tf.keras.Sequential([
    preprocessor,
    tf.keras.layers.RandomFlip('horizontal'),                         
    tf.keras.layers.RandomRotation(0.2),
    base_model_layer,                    
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2),
])

model4.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model_2 (Functional)        (None, 160, 160, 3)       0         
                                                                     
     random_flip_3 (RandomFlip)  (None, 160, 160, 3)       0         
                                                                     
     random_rotation_3 (RandomRo  (None, 160, 160, 3)      0         
     tation)                                                         
                                                                     
     model_1 (Functional)        (None, 5, 5, 1280)        2257984   
                                                                     
     dropout_4 (Dropout)         (None, 5, 5, 1280)        0         
                                                                     
     flatten_2 (Flatten)         (None, 32000)             0         
                                                                     
     dense_4 (Dense)             (None, 64)                2048064   
                                                                     
     dense_5 (Dense)             (None, 2)                 130       
                                                                     
    =================================================================
    Total params: 4,306,178
    Trainable params: 2,048,194
    Non-trainable params: 2,257,984
    _________________________________________________________________


Okay wow! So there is a lot of complexity hidden in base_model_layer as we need to train 2,048,194 parameters. Let's get to it then!


```python
model4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 11s 109ms/step - loss: 0.0651 - accuracy: 0.9835 - val_loss: 0.1463 - val_accuracy: 0.9790
    Epoch 2/20
    63/63 [==============================] - 7s 95ms/step - loss: 0.0894 - accuracy: 0.9740 - val_loss: 0.0920 - val_accuracy: 0.9802
    Epoch 3/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.0676 - accuracy: 0.9760 - val_loss: 0.1562 - val_accuracy: 0.9703
    Epoch 4/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1106 - accuracy: 0.9805 - val_loss: 0.1298 - val_accuracy: 0.9827
    Epoch 5/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.1482 - accuracy: 0.9720 - val_loss: 0.0999 - val_accuracy: 0.9827
    Epoch 6/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1941 - accuracy: 0.9690 - val_loss: 0.0883 - val_accuracy: 0.9802
    Epoch 7/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1216 - accuracy: 0.9705 - val_loss: 0.1050 - val_accuracy: 0.9814
    Epoch 8/20
    63/63 [==============================] - 7s 99ms/step - loss: 0.0780 - accuracy: 0.9765 - val_loss: 0.0976 - val_accuracy: 0.9839
    Epoch 9/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0588 - accuracy: 0.9835 - val_loss: 0.0900 - val_accuracy: 0.9827
    Epoch 10/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0415 - accuracy: 0.9885 - val_loss: 0.0348 - val_accuracy: 0.9913
    Epoch 11/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0517 - accuracy: 0.9855 - val_loss: 0.0438 - val_accuracy: 0.9864
    Epoch 12/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0508 - accuracy: 0.9815 - val_loss: 0.0590 - val_accuracy: 0.9814
    Epoch 13/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0422 - accuracy: 0.9860 - val_loss: 0.0693 - val_accuracy: 0.9827
    Epoch 14/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.0301 - accuracy: 0.9890 - val_loss: 0.0676 - val_accuracy: 0.9802
    Epoch 15/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.0278 - accuracy: 0.9910 - val_loss: 0.0493 - val_accuracy: 0.9876
    Epoch 16/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.0252 - accuracy: 0.9895 - val_loss: 0.0853 - val_accuracy: 0.9777
    Epoch 17/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0254 - accuracy: 0.9915 - val_loss: 0.0548 - val_accuracy: 0.9827
    Epoch 18/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.0310 - accuracy: 0.9890 - val_loss: 0.0803 - val_accuracy: 0.9839
    Epoch 19/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.0271 - accuracy: 0.9885 - val_loss: 0.0887 - val_accuracy: 0.9790
    Epoch 20/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.0391 - accuracy: 0.9870 - val_loss: 0.1019 - val_accuracy: 0.9790


By incorporating the base_model_layer into our model, we were able to obtain a validation accuracy of approximately 98%, so we are fairly pleased! We have come a long way since our baseline model which had a validation accuracy of 50%. While overfitting is always a concern, it appears that we have done our best to minimize it in this model as the training and validation accuracy stabilized to be very similar values.


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f502579d3d0>




    
![image-example.png](/images/output_45_1.png)

    


When we evaluate model4 on our testing data, we obtain an accuracy of 97%. This means that our model was able to correctly differentiate between cats and dogs 97% of the time, so we are pretty happy with these results!


```python
model4_score = model4.evaluate(test_dataset)
```

    6/6 [==============================] - 1s 73ms/step - loss: 0.1609 - accuracy: 0.9583


### Score on Test Data


Let's take one last look at the testing accuracy that we were able to obtain from our models!


```python
print("model 1 score: " + str(round(model1_score[1],2)))
print("model 2 score: " + str(round(model2_score[1],2)))
print("model 3 score: " + str(round(model3_score[1],2)))
print("model 4 score: " + str(round(model4_score[1],2)))

```

    model 1 score: 0.56
    model 2 score: 0.53
    model 3 score: 0.77
    model 4 score: 0.96


Thank you for reading! I'll see you next week for our last blog post.
~ Emma
