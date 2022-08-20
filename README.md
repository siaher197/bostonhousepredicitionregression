**Boston House Prediciton using  ML Regression**

In this project, we will develop and evaluate the performance and the predictive power of a model trained and tested on data collected from houses in Boston’s suburbs.
A model like this would be very valuable for a real state agent who could make use of the information provided in a dayly basis.
Getting the Data and Previous Preprocess
The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston.

The features can be summarized as follows:

CRIM: This is the per capita crime rate by town
ZN: This is the proportion of residential land zoned for lots larger than 25,000 sq.ft.
INDUS: This is the proportion of non-retail business acres per town.
CHAS: This is the Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
NOX: This is the nitric oxides concentration (parts per 10 million)
RM: This is the average number of rooms per dwelling
AGE: This is the proportion of owner-occupied units built prior to 1940
DIS: This is the weighted distances to five Boston employment centers
RAD: This is the index of accessibility to radial highways
TAX: This is the full-value property-tax rate per $10,000
PTRATIO: This is the pupil-teacher ratio by town
B: This is calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African American descent by town
LSTAT: This is the percentage lower status of the population
MEDV: This is the median value of owner-occupied homes in $1000s

Data Exploration
In the first section of the project, we will make an exploratory analysis of the dataset and provide some observations.

Calculate Statistics

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

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

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




