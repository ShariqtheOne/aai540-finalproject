# aai540-finalproject Bank Churn Prediction

## Project Summary

We built and deployed an ML system for churn prediction using a bank customer dataset. The objective is to predict whether a customer is likely to exit or not based on various features such as credit score, geography, gender, age, tenure, balance, and the number of products used. This is a binary classification problem.

## Project Team
- <ins>Members</ins>: Joseph Binny, Abdul Shariq, Viktor Veselov
- <ins>Program</ins>: University of San Diego, Master of Science: Applied Artificial Intelligence
- <ins>Professor</ins>: Mark Christenson
- <ins>Term</ins>: Spring 2024

## Brief Outline
In this project we leverage AWS SageMaker to deploy an end-to-end ML system including creating a data lake, feature store, deploy a monitor to SageMaker Endpoint with a monitoring schedule, and implementing a CICD pipeline to train a model and invoke it for a batch-transform job.

Below is the CICD pipeline architecture  

<img src="./pictures/pipeline-design.png" />
