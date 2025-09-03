# Diabetes Prediction using Machine Learning

This project aims to predict the likelihood of a patient having diabetes based on various health-related features. We experimented with several machine learning models to find the most accurate predictor.

## Project Overview

The primary goal of this project is to build a highly accurate classification model for diabetes prediction. We explored and evaluated five different algorithms on a comprehensive diabetes dataset. Our analysis showed that the XGBoost Classifier provided the highest accuracy among the tested models.

## Dataset

The dataset used for this project is the "Diabetes Prediction Dataset" available on Kaggle. It contains several medical predictor variables and one target variable, diabetes.

**Dataset Link:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Models and Results

We implemented and evaluated the performance of the following five machine learning models:

* Convolutional Neural Network (CNN)
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Random Forest
* XGBoost Classifier

After rigorous training and testing, the models yielded the following accuracies:

| Model                | Accuracy |
| -------------------- | -------- |
| XGBoost Classifier   | **97.07%**   |
| Random Forest        | 96.27%   |
| CNN                  | 96.46%   |
| Logistic Regression  | 96.55%   |
| K-Nearest Neighbors (KNN) | 96.22% |

The XGBoost model emerged as the top-performing model with an impressive accuracy of 97.07%.

## Technologies Used

* **Programming Language:** Python
* **Libraries:**
    * Pandas
    * NumPy
    * Scikit-learn
    * XGBoost
    * TensorFlow / Keras (for CNN)
    * Matplotlib / Seaborn (for visualization)

## How to Run the Project

To replicate the results, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Priyangshu713/diabetes_prediction.git](https://github.com/Priyangshu713/diabetes_prediction.git)
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the dataset from the link provided above and place it in the project's `data` directory.

4.  Run the Jupyter Notebook or Python script to see the model training, evaluation, and results.

## Conclusion

This project successfully demonstrates the effectiveness of machine learning, particularly the XGBoost algorithm, in the medical field for predicting diabetes. An accuracy of 97% shows great promise for its use as a preliminary diagnostic tool.
