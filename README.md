# DS-Minor Max de Goede
Work from Max de Goede on the Data Science Minor. Here you will find a list of my findings on the following:
## 1. ML (Machine Learning) Introduction
The Machine Learning introduction was a recap of what was made in the previous Applied Data Science courses (ADS-A/ADS-B).

Here we looked at the knowledge that we had already learned and worked with it again.

In the most basic form Machine Learning is teaching a computer how to do tasks on its own by training it. There are two major types of algorithms: supervised and unsupervised:
### Supervised learning
When we look at supervised learning there are a few things that are important. First being that the network is trained on the input and out put of the network. This means that the network is told when it predicted something wrong. 

Some examples of supervised learning:
1. Decision Tree
2. K-nearest neighbor:
  - Looks at the x nearest data points and classifies a datapoint based on its neighbors, below you can see how this works![image KNN](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/03/knn3.png)
3. Neural Network
4. Support Vector Machine

### Unsupervised learning
The goal of unsupervised learning is finding transformations of the input data without the support of any labelled output. In other words: the machine learning model is trained with unlabelled data. Hence, the algorithm tries to identify patterns in the data and give a response.

An example of Unsupervised learning:
1. k-means clustering

### What did i do?
For the exercise we had t work with the dataset. transform the data so we could take it make use of it and and traing a network on it. The most notable examples of this was using PCA to clean te data and seeing what effects it had on the result of the network.
## 2. ANN (Artificial Neural Network)
### What is an ANN?

Artificial Neural Networks (ANN) are computing systems inspired by biological neural networks that form human or animal brains. An ANN typically consists of multiple layers of neurons. All neurons in one layer are connected to all neurons in the next layer. Such a neural network is also called a multilayer perceptron. In the image below you can see a neural network.

![Image of Artificial Neural Network](https://miro.medium.com/max/2500/1*ZB6H4HuF58VcMOWbdpcRxQ.png)

### What did i do?
For the artificial neural network exercise i used a dataset from Mnist. Here i used the fashion Mnist dataset and tried to create a network that would predict the clothing. The main goal here was to create my first networks using the Tensorflow and Keras libraries. Here i played with the hyperparameters such as:
1. Hidden layers
2. Learning rate
3. Epochs

![Image of the Fashion Mnist Dataset](https://miro.medium.com/max/3200/1*QQVbuP2SEasB0XAmvjW0AA.jpeg)
Fashion Mnist Dataset
## 3. CNN (Convolutional Neural Network) 1 - Standing on the shoulders of giants
### What is an CNN?
CNNs are mainly used for image analysis, in particular image classification. The difference between a multilayer perceptron and a CNN is that CNNs contain a certain type of hidden layers, convolutional layers, that are aimed at detecting patterns. These patterns may by relatively simple patterns such as edges or more complicated patterns such as faces. Simple patterns can by detected by filtering images using convolution. Complicated patterns are being detected using multiple convolutional layers. The result of these convolutional layers are then fed as input to a regular (multilayer) perceptron for further analysis and classification.
### What did i do?
For the first CNN exercise we looked at existing netowrks and tried applying these weights and to use transfer learning. Personally i made a network using Mask-Rcnn and applied weights that were already pre-trained (coco_weights). The goal here was to train a network to detect and recognize ships. For this i had to label the ships on the image with Max/Min Y/X.
![Image Convolutional Neural Network](https://miro.medium.com/max/2510/1*vkQ0hXDaQv57sALXAJquxA.jpeg)
## 4. CNN 2
## 5. RL (Reinforcement Learning) 
## 6. NLP
## 7. Project
