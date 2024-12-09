## Project Name : Image Classification using Caltech-101
- **Contributors :** Shravani, Shireesha and Rashmi
- **Date :** Dec 17, 2023

### Project Overview
- The project aims to create a system for recognizing and categorizing images into 101 predefined groups using the Caltech-101 dataset. 
The team plans to use the VGG16 architecture, a type of convolutional neural network (CNN), and apply transfer learning to achieve this goal. VGG16 is selected for its efficiency in extracting important features from images.

### About type of AI system
- **Neural Network Classifier**
  - The Neural network classifier, once trained, serves as a specialized neural network designed for classification tasks. Its structure involves a sequence of interconnected layers, starting with the first fully connected layer, 
  which directly takes input from the network. Each subsequent layer is connected to the output of the previous layer. In each fully connected layer, the input undergoes a transformation: it is multiplied by a weight matrix and then has a bias vector added to it. 
  Following each of these operations is an activation function, contributing to the network's ability to capture complex patterns. The last fully connected layer, coupled with a softmax activation function, plays a crucial role in generating the network's output. 
  This final stage produces classification scores and predicts the corresponding labels for the input data. In essence, it transforms the raw network output into a meaningful classification result.(MATLAB, Accessed December 17, 2023)

### About Dataset
- **Dataset Name:** caltech-101
- **Authors:** Fei-Fei Li, Marco Andreetto and Marc'Aurelio Ranzato
- **Dataset created:** September 8, 2022
- **Dataset modified:** February 21, 2023
- Caltech-101 is a dataset with pictures of different objects grouped into 101 categories, plus one for background clutter. Each picture is labeled with a single object, and there are about 9,000 images in total. 
The classes have varying numbers of images, and the pictures come in different sizes, usually with edges between 200 and 300 pixels.(TensorFlow, December 17, 2023) 

### About AI system
- This project is about the image classification which uses the VGG16 architecture and caltech-101 dataset. 
The VGG16 architecture is characterized by a deliberate design choice, emphasizing simplicity with a focus on 3x3 convolutional layers using a stride of 1 and always employing same padding. 
The consistent pattern of convolution layers followed by 2x2 max-pooling layers with a stride of 2 is maintained throughout the entire structure. The term "16" in VGG16 denotes the presence of 16 layers with trainable weights. 
The network concludes with two fully connected layers and a softmax activation for output. VGG16 is a substantial model, boasting around 138 million parameters.(Naidu, Ashwin. (2019) 2023)


### Reference:
- “Neural Network Model for Classification - MATLAB.” n.d. Accessed December 17, 2023. https://www.mathworks.com/help/stats/classificationneuralnetwork.html.
- “Caltech101 | TensorFlow Datasets.” n.d. TensorFlow. Accessed December 17, 2023. https://www.tensorflow.org/datasets/catalog/caltech101.
- Naidu, Ashwin. (2019) 2023. “Ashushekar/VGG16.” Python. https://github.com/ashushekar/VGG16.




