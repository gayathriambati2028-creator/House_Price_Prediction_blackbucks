The dataset contains information such as:

* City
* Area size
* Number of rooms
* Furnishing type
* Crime rate
* Market price

The main goal is to predict the price of a house based on its features.

The data is cleaned before training the model.

Text data such as **City, Locality, and Furnishing** are converted into numerical values using **One-Hot Encoding (`pd.get_dummies`)**.

One-Hot Encoding is used to convert categorical (text) data into numerical (0 and 1) values so that machine learning algorithms can understand and use the data.

The dataset is divided into **80% training data** and **20% testing data**.

A **Random Forest Regression** model from **Scikit-learn** is used to train the data.

The trained model predicts house prices using the testing data.

The model performance is evaluated using:

* **Mean Absolute Error (MAE)** – measures the average prediction error.
* **R² Score** – measures how well the model predicts house prices.

**Feature Importance** is also calculated to identify which features have the greatest impact on predicting house prices. It ranks all the input features based on their importance, helping to understand which factors influence the house price the most.

**Main Function**

The main() function controls the complete workflow of the House Price Prediction System. It calls all the required functions in the correct order, including loading the dataset, selecting and cleaning the data, encoding categorical values, splitting the dataset, training the Random Forest model, evaluating its performance, and displaying the feature importance. The statement `if __name__ == "__main__":` ensures that the main() function runs only when the program is executed directly.

