# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Our main task in this prpject is to write clean and organized code that follows the best coding practices. What we're trying to do in this project is predict when customers might stop using a bank's services, like when they close their accounts or switch to another bank.

Here's what we're doing in simple terms:

**Explore the Data:** We start by looking at the information we have in our dataset, which is like a big collection of data. We want to understand it better before doing anything else.

**Prepare the Data:** To make our predictions work well, we need to get the data in good shape. We organize and change it a bit to create 19 different pieces of information that our prediction tools can use.

**Train the Models:** We're using two different methods to make predictions, like using two different types of tools. One is called "sklearn random forest," and the other is "logistic regression." We're testing both to see which one works better.

**Figure out Important Information:** We want to know which things are the most important for making predictions. We use a tool called the SHAP library to help us see which things have the biggest impact.

**Save the Best Models:** Once we figure out the best way to make predictions, we save those methods along with how well they work. This way, we can use them again later.

Also, we've made sure our code follows the rules for good coding. We formatted our code to meet the PEP8 standard, and it got a high score of over 9.4 for clean code from the pylint module. This shows that our code is well-organized and easy to read.
## Files and data description

**Folders**

**Data:** Contains the dataset in CSV format.
images
eda: Contains the output of data exploration.
results: Stores testing results.
**models:** Contains saved models in .pkl format.
**logs:** Holds logs generated during the testing of the churn_library.py file.

**Project Files**

**churn_library.py:** This is a Python script file.
**churn_notebook.ipnyb:** This is a Jupyter Notebook file.
**requirements.txt:** This file lists the required dependencies for the project.

**Pytest Files**
**test_churn_script_logging_and_tests.py:** This file contains unit tests and configuration files for testing.
 

## Running Files
To successfully run the project, ensure you have Python 3.8 installed, along with the necessary Python packages as specified in the requirements.txt file.
To initiate the project, execute the script python churn_library.py from the project's root folder.

The functionality of the project script, churn_library.py, has been thoroughly tested using the pytest Python package.

To run unit tests, simply type pytest in the command line from the project's main folder.
These tests will automatically assess the project functions and generate a log file in the "logs" folder.



