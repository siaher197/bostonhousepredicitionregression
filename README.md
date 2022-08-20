**Boston House Prediciton using  ML Regression**

In this project, we will develop and evaluate the performance and the predictive power of a model trained and tested on data collected from houses in Boston’s suburbs.A model like this would be very valuable for a real state agent who could make use of the information provided in a dayly basis.
Getting the Data and Previous Preprocess The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston.

Project Highlights
This project is designed to get you acquainted to working with datasets in Python and applying basic machine learning techniques using NumPy and Scikit-Learn. Before being expected to use many of the available algorithms in the sklearn library, it will be helpful to first practice analyzing and interpreting the performance of your model.Things you will learn by completing this project:
How to use NumPy to investigate the latent features of a dataset.
How to analyze various learning performance plots for variance and bias.
How to determine the best-guess model for predictions from unseen data.
How to evaluate a model's performance on unseen data using previous data.

**Data Exploration**
In the first section of the project, we will make an exploratory analysis of the dataset and provide some observations.

**Calculate Statistics**

# Minimum price of the data
minimum_price = np.amin(prices)
# Maximum price of the data
maximum_price = np.amax(prices)
# Mean price of the data
mean_price = np.mean(prices)
# Median price of the data
median_price = np.median(prices)
# Standard deviation of prices of the data
std_price = np.std(prices)

![statistics](https://user-images.githubusercontent.com/109465506/185759230-dae8a091-ff36-41b0-a151-e62fa32c85d9.png)

**Exploratory Data Analysis**
Scatterplot and Histograms
We will start by creating a scatterplot matrix that will allow us to visualize the pair-wise relationships and correlations between the different features.
It is also quite useful to have a quick overview of how the data is distributed and wheter it cointains or not outliers.
We can spot a linear relationship between ‘RM’ and House prices ‘MEDV’. In addition, we can infer from the histogram that the ‘MEDV’ variable seems to be normally distributed but contain several outliers.

![histogram](https://user-images.githubusercontent.com/109465506/185759311-49b8b06f-3762-47d6-9876-b0074b93eb4e.png)
 
** Correlation Matrix**

We are going to create now a correlation matrix to quantify and summarize the relationships between the variables.
This correlation matrix is closely related witn covariance matrix, in fact it is a rescaled version of the covariance matrix, computed from standardize features.
It is a square matrix (with the same number of columns and rows) that contains the Person’s r correlation coefficient.
To fit a regression model, the features of interest are the ones with a high correlation with the target variable ‘MEDV’. From the previous correlation matrix, we can see that this condition is achieved for our selected variables.

![corr matrix](https://user-images.githubusercontent.com/109465506/185759366-656175c5-d615-45dc-b484-1dda8a40a79e.png)

*Analyzing Model’s Performance*
Learning Curves
The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased.Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R2, the coefficient of determination.
# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)
![decision tree regression crve](https://user-images.githubusercontent.com/109465506/185760564-2c85951b-df4e-4402-a5ae-c63d41679456.png)

Complexity Curves
The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation.
Similar to the learning curves, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the performance_metric function.
# Produce complexity curve for varying training set sizes and maximum depths
vs.ModelComplexity(X_train, y_train)
![desicion tree complexity](https://user-images.githubusercontent.com/109465506/185760688-15c4a2bd-3732-4bb2-938c-6e1791318017.png)

**Bias-Variance Tradeoff**





