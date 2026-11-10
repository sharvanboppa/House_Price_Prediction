# 🏡 House Price Prediction Using Machine Learning

## 📌 Project Overview

House Price Prediction is a Machine Learning web application that predicts the estimated price of a house based on various property features.

The project uses **Linear Regression**, a supervised Machine Learning algorithm, to analyze housing data and estimate property prices. A user-friendly web interface is developed using Flask, HTML, CSS, and Python.

Users can enter house details such as location, number of bedrooms, bathrooms, living area, and year of construction to get an estimated house price.

## ✨ Features

- Predicts house prices using Machine Learning.
- User-friendly and responsive web interface.
- Accepts multiple property details as input.
- Supports city and location-based predictions.
- Displays estimated house prices instantly.
- Uses a trained Machine Learning model for predictions.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| Linear Regression | Prediction Algorithm |
| Flask | Web Framework |
| HTML & CSS | Frontend Development |
| Pickle | Model Serialization |

## 📂 Project Structure

```text
House_Price_Prediction/
│
├── dataset/
│   └── data.csv
│
├── static/
│   └── CSS and other static files
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── house_price_model.pkl
├── requirements.txt
├── Procfile
└── README.md
```

## ⚙️ Input Features

The model uses the following property details:

- Bedrooms
- Bathrooms
- Living Area (Square Feet)
- Lot Area (Square Feet)
- Number of Floors
- House Condition
- Year Built
- City
- State ZIP Code

## 🤖 Machine Learning Algorithm

This project uses **Linear Regression** to predict house prices.

### Model Development Steps

1. Load the housing dataset.
2. Perform data cleaning and preprocessing.
3. Remove missing and invalid values.
4. Convert categorical features into numerical values using One-Hot Encoding.
5. Split the dataset into training and testing sets.
6. Train the Linear Regression model.
7. Evaluate the model using R² Score and RMSE.
8. Save the trained model using Pickle.

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/sharvanboppa/House_Price_Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd House_Price_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Machine Learning Model (Optional)

```bash
python train_model.py
```

This generates the trained model file `house_price_model.pkl`.

### 7. Run the Flask Application

```bash
python app.py
```

### 8. Open in Browser

Visit:

```text
http://127.0.0.1:5000/
```

Enter the required house details and click the prediction button to view the estimated house price.

## 📊 Model Evaluation

The model performance is evaluated using:

- **R² Score:** Measures how well the model explains variations in house prices.
- **RMSE:** Measures the prediction error of the model.

## 🎯 Project Objective

The main objective of this project is to apply Machine Learning concepts to a real-world problem and develop a web-based application that helps users estimate house prices based on property characteristics.

## 👨‍💻 Author

**Sharvan Boppa**

GitHub: [@sharvanboppa](https://github.com/sharvanboppa)

## ⭐ Support

If you find this project useful, consider giving this repository a star!

---

**Built with Python, Machine Learning, and Flask.**
