# Neural Network with MLOps using GitHub Actions

This project showcases the integration of Machine Learning Operations (MLOps) with neural network models using GitHub Actions for continuous integration and continuous deployment (CI/CD). The primary focus is to create a neural network to classify MNIST dataset digits using the PyTorch framework and demonstrate automated testing and deployment pipelines.

## Table of Contents 
- [Features](#Features) 
- [Objectives](#Objectives) 
- [Directory Structure](#Directory-Structure) 
- [Technologies Used](#Technologies-Used)

## Features
- **Neural Network Model**: Build and train a neural network model on the MNIST dataset using PyTorch.
- **MLOps Integration**: Implement MLOps practices to streamline the machine learning lifecycle, from development to deployment.
- **GitHub Actions**: Utilize GitHub Actions for CI/CD to automate testing, validation, and deployment of the model.
- **Automated Testing**: Create comprehensive test cases to ensure model accuracy and reliability throughout the deployment pipeline.
- **Reproducibility**: Ensure reproducibility of results with version-controlled code and dependencies.

## Objectives
- Build and Train Model: Develop a neural network model capable of recognizing handwritten digits from the MNIST dataset.
- Automate Workflows: Set up GitHub Actions workflows to automate model training, testing, and deployment processes.
- Demonstrate MLOps: Exhibit the benefits of MLOps by integrating CI/CD practices into the machine learning development cycle.

## Directory Structure
<pre>
.  
├── .github  
│   └── workflows  
│       └── main.yml           # GitHub Actions workflow file  
├── train.py                   # Script for data loading, model definition, training, and validation  
├── test_model.py              # Script for testing the model  
├── requirements.txt           # Python dependencies  
└── README.md                  # Project README file  
</pre>

## Technologies Used
> - **PyTorch**: For building and training the neural network model.
> - **MNIST Dataset**: As the primary data source for training and testing the model.
> - **GitHub Actions**: For implementing CI/CD workflows.
> - **Python**: As the primary programming language for the project.
