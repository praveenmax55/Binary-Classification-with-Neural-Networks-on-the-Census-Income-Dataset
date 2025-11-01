# Census Income Classification using PyTorch

## AIM
To build a binary classification model using the Census Income dataset to predict whether an individual earns more than $50,000 annually.

## THEORY
Classification is a supervised learning task where the goal is to predict categorical outcomes based on input features. Traditional models often struggle with heterogeneous data (mix of categorical and continuous). Neural networks with embeddings and normalization can handle such complexity efficiently. In this experiment, categorical variables are represented using embeddings, and continuous variables are normalized. A feedforward neural network is trained to classify income levels.

## MODEL DESIGN
# Input Features
Categorical: sex, education, marital-status, workclass, occupation

Continuous: age, hours-per-week

Neural Network Architecture

Embedding layers for categorical inputs

Batch normalization for continuous inputs

One hidden layer with 50 neurons, dropout = 0.4

Output layer with 2 classes (<=50K or >50K)

## DESIGN STEPS
# STEP 1: Data Preparation
Load dataset (income.csv, 30,000 rows).

Separate categorical, continuous, and label columns.

Convert values into arrays and PyTorch tensors.

Split dataset into training (25,000) and testing (5,000).

# STEP 2: Model Definition
Define a custom TabularModel class.

Combine categorical embeddings with continuous features.

Apply batch normalization, linear layers, ReLU activation, and dropout.

# STEP 3: Loss Function and Optimizer
Use nn.CrossEntropyLoss() for classification.

Use torch.optim.Adam optimizer with learning rate 0.001.

# STEP 4: Training
Train for 300 epochs.

Record training loss and accuracy.

# STEP 5: Evaluation
Evaluate model on the test set.

Report final loss and accuracy.

# STEP 6: User Input Prediction (Bonus)
Function accepts user details such as age, sex, education, hours/week.

Model outputs predicted income class.

# PROGRAM AND REQUIREMENTS:


## RESULT
Thus, a tabular neural network model was successfully developed using PyTorch to classify income level. The model achieved ~85% test accuracy.
