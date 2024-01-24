# MachineLearning
- Consist of Linear Regression Models
    - pStock.py consists of a multilinear regression model (Prediction of closing stock prices based on open, close, high, low variables)
    - Uses Sklearn to preprocess datasets, produce models and to test models 
    - Sklearn metrics used to determine accuracy of the dataset

- KNNeighbour Recognition Models
    - Based on features of images the model will classify the images using the target labels 
    - Nearest n_neighbours (data points are used) to classify 
    - Sensitive model to noise and outliers (stores dataset: recall prevalent)

- Decision Tree Classifiers
    - Use regression values
    - Each node is a decision and explicit training 

Standardisation
    - This is heavily required to prevent noise in datasets e.g., wrong scales used to measure data.
    - Standardisation therefore ensures the scales and metrics

Predictions:
    - Coefficient Values and N_neighbours 

Evaluation:
    - Dataset Size (Larger the dataset the better, Smaller can lead to higher accuracy)
    - Overfitting 
    - F1 Scores
    - Recall Metrics
    - Precision
    - Mean Squared Metrics (MSE)