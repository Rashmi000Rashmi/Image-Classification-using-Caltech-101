### Project Title: Image Classification using Caltech101.

**Name of the Document** : Design of Image Classification

**Authors** : Shravani, Shireesha and Rashmi.

**Date** : 19-11-2023


#### AI System Overview

    --Author: Shireesha, Reviewer : Shravani, Rashmi

Our Team selected to Investigate on `Neural Network Classifier`. A neural network classifier is a type of artificial neural network that consists of functions called parameters, 
“which allows the computer to learn and fine tune itself by analyzing new data”.(Knocklein Oliver, 2019) A neural network consists of layers of interconnected neurons. 
These layers include an input layer, one or more hidden layers and an output layer.(Rohini G, 2021)

We have chosen `VGG16 pre-trained model` which is a type of `Convolutional neural network`. The model is based on the VGG16 has a straightforward and symmetric architecture, 
which is composed of a series of convolutional and pooling layers followed by fully connected layers. “VGG uses a smaller convolutional filter, 
which reduces the network’s tendency to over-fit during training exercises. A 3×3 filter is the optimal size because a smaller size cannot capture left-right and up-down information. 
Thus, VGG is the smallest possible model to understand an image’s spatial features. Consistent 3×3 convolutions make the network easy to manage.”(“Understanding VGG16: Concepts, Architecture, and Performance.”, November 20, 2023 )

Using a pretrained model like VGG16 on the `Caltech101 dataset` is like starting with a neural network that has already learned a lot about images from a large dataset. 
This pretrained model, VGG16, has figured out how to recognize different patterns and objects. Instead of training a new model from scratch on Caltech101, we use VGG16 as a starting point. 
We adjust its weights slightly to adapt to the specifics of Caltech101. This approach saves time, resources, and leads to better performance on the smaller dataset.

#### Relevant Theoretical Background

    --Author : Shravani, Reviewer : Shireesha, Rashmi

The type of project we selected is `neural network classifier`, it is trained to produce the desired outputs, and different models are used to predict future results with the data. Neural network has three different layers such as `Input layer`, `Hidden layer`, `Output layer`, and each layer can perform a specific task or function. 
The input layer is composed not of full neurons, but rather consists simply of the record's values that are inputs to the next layer of neurons. 
The next layer is the hidden layer. Several hidden layers can exist in one neural network. The final layer is the output layer, where there is one node for each class. A single sweep forward through the network results in the assignment of a value to each output node, and the record is assigned to the class node with the highest value. **(Priya Pedamkar, April 15, 2019) and (Neural Network Classification, March 23, 2012)**

The neural network model that team has selected is `VGG16` which is specific `convolutional neural network` architecture. VGG16 uses convolution layers with a 3x3 filter and a stride 1 that are in the same padding and maxpool layer of 2x2 filter of stride 2. 
It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has two fully connected layers, followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network, and it has about 138 million (approx) parameters. **(Rohit Thakur, Nov 20, 2023)**

Need to have a full understanding of the dataset i.e., caltech-101 such as the total no.of categories(101) and total no.of images(9146) which are of 200*300 pixels. Knowledge on libraries like `Tensorflow` - used to design the model, `Numpy` - used for processing image data stored as arrays and transforming the categorical labels into a binary format, `Scikit-Learn` - used to split the data and to create training and testing datasets, `Keras`, Data Augmentation and how they can improve model generalization. 
Understanding of basic concepts such as supervised learning, classification, and model evaluation. Skills in preparing and preprocessing image data and also ability to train a model using labeled datasets and evaluate its performance. Understanding the concept of loss functions used to measure the model's error during training.
Familiarity with metrics like accuracy, precision, recall and F1 score for assessing model performance.

#### AI System Development and Evaluation

    --Author : Rashmi, Reviewer : Shravani, Shireesha


**Build an Image Classification System using Caltech101 Dataset**

      Objective : The project focused on developing an AI model
                  that can categorize images into different categories.
                  In this project, a pre-trained VGG16 model will be used,
                  and its ability to identify images from the Caltech101 dataset will be evaluated.

      Development: The project code analyzes the images, divides them into training and test sets 
                  (`X_train`, `X_test`, `y_train`, `y_test`), and builds a AI
                   model with VGG16 as the base and extra layers (`Flatten`, `Dense`, `Dropout`)
                   and a final Dense layer with a `softmax activation` function for different class classification.
                   Using the `train_test_split` functio from `Scikit-learn`, 
                   the data is divided into training and testing sets

      Evaluation: Training the model on the training set and testing it on the distinct,
                  unseen test set allows to review the model's performance.
                  Evaluation metrics are computed to assess the system's classification 
                  performance, including accuracy and loss.

**Overfitting Analysis of the AI system**

     Objective : Examine how the AI model behaves and performs using the training and testing
                 datasets to find any overfitting occurrences.

     Development: `ImageDataGenerator` uses data augmentation approaches during training
                  to improve generalization and reduce overfitting which helps
                  the model learn stable and distinct features from the existing data by producing different
                  versions of the images. the project code maintains track of the AI model's
                  performance (accuracy and loss) through several epochs on both training and test sets.

     Evaluation: Plotting the accuracy and loss graphs from training allows us to see the overfitting behavior.
                 Finding overfitting or underfitting tendencies can be facilitated by comparing 
                 training and accuracy performance.

**Improve the AI system's Classification Accuracy by Training It**

     Objective : Increase the classification accuracy of the AI system by feeding the model more data to train on.

     Development: To achieve high accuracy in categorizing images into their respective categories,
                  the AI system is trained using the training dataset  (`X_train`, `y_train`) to identify characteristics
                  and patterns within the images.

     Evaluation: The performance of our AI model is evaluated using its accuracy and loss
                 measures on the test dataset. High accuracy on the test set is the goal, 
                 as this shows how effective our AI model is to the new data images or unseen data.

**Performance of the AI Model During Training**

     Objective : As the AI model gains knowledge from the training set, track and evaluate
                 the variations in accuracy and loss.

     Development: To visualize the AI model's learning process and performance on both training and test sets,
                  the code will provide a plot on accuracy and loss graph.

     Evaluation: The graphical depiction helps in understanding the accuracy and loss trend through epochs,
                  offering insights into the learning behavior of the model and possible problems such as overfitting.

#### Project Milestones and Deliverables

    --Authors : Shireesha, Shravani, Rashmi

| Task                                                            | Description                                                              | Status    | Assigned to                 | start date   | End Date     |
|-----------------------------------------------------------------|--------------------------------------------------------------------------|-----------|-----------------------------|--------------|--------------|
| **Initiation:**                                                 |                                                                          |           |                             | Oct 31, 2023 | Nov 06, 2023 |
| Project and Dataset Selection                                   | Selection of Type of AI System and Dataset                               | Completed | Rashmi, Shireesha, Shravani |              |              |
| Finalize project proposal                                       | About type of Project, AI system and Tools                               | Completed | Rashmi, Shireesha, Shravani |              |              |
| **Design:**                                                     |                                                                          |           |                             | Nov 19, 2023 | Nov 20, 2023 |
| Design Document                                                 | Overview of the Project and its Objectives                               | Completed | Rashmi, Shireesha, Shravani |              |              |
| **Code:**                                                       |                                                                          |           |                             | Nov 21, 2023 | Dec 05, 2023 |
| Load the data presplit for test                                 | Load the Train and test data for preprocessing                           | To do     | Shravani                    |              |              |
| Design the supervised learning model                            | Define the VGG16 model Architecture                                      | To do     | Shireesha                   |              |              |
| Train the neural network to correctly classify the test dataset | Train the model to predict and classify the images                       | To do     | Shravani, Shireesha, Rashmi |              |              |
| **Testing:**                                                    |                                                                          |           |                             | Dec 06, 2023 | Dec 10, 2023 |
| Overfit the neural network                                      | Examine the model's performance on both the training and validation sets | To do     | Rashmi                      |              |              |
| Graph the model performance as it overfits the neural network   | Models performance graph incase of overfitting                           | To do     | Rashmi, Shireesha, Shravani |              |              |                                                   |                                                |             |                             |              |              |
| **Documentation:**                                              |                                                                          |           |                             | Dec 11, 2023 | Dec 18, 2023 |
| Project final Report                                            | Documentation of Model's performance                                     | To do     | Shireesha, Shravani, Rashmi |              |              |


#### References

    --Authors : Shireesha, Shravani, Rashmi

Knocklein, Oliver. 2019. “Classification Using Neural Networks.” Medium. June 15, 2019. https://towardsdatascience.com/classification-using-neural-networks-b8e98f3a904f.
  
Rohini G. 2021. “Everything You Need to Know about VGG16.” Medium (blog). September 23, 2021. https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918.
                                            
“Understanding VGG16: Concepts, Architecture, and Performance.” n.d. Datagen. Accessed November 20, 2023. https://datagen.tech/guides/computer-vision/vgg16/.

Priya Pedamkar, “What Are Neural Networks? | Explanation, History & Career.” 2019. EDUCBA (blog). April 15, 2019. https://www.educba.com/what-is-neural-networks/.

“Neural Network Classification.” 2012. Solver. March 23, 2012. https://www.solver.com/xlminer/help/neural-networks-classification-intro.

Rohit Thakur, “Beginners Guide to VGG16 Implementation in Keras | Built In.” n.d. Accessed November 20, 2023. https://builtin.com/machine-learning/vgg16.

**Code References:**

Brownlee, Jason. 2017. “How to Use The Pre-Trained VGG Model to Classify Objects in Photographs.” MachineLearningMastery.Com (blog). November 7, 2017. https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/.

Truong, Dang Manh. 2019. “Keras Image Classification: High Accuracy Shown but Low on Test Images.” Forum post. Stack Overflow. https://stackoverflow.com/q/58344477.