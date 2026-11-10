# 🏠 House Price Prediction

## 📌 Project Overview

House Price Prediction is a Machine Learning web application that predicts house prices based on various property features using **Multiple Linear Regression**.

The application allows users to enter property details such as bedrooms, bathrooms, living area, lot size, floors, condition, year built, city, and ZIP code to estimate the house price.

The project uses Python for machine learning, Flask for web development, and HTML/CSS for the user interface.

## 🎯 Project Objectives

- Analyze historical house sales data.
- Perform data preprocessing and feature engineering.
- Build a Multiple Linear Regression model.
- Predict house prices based on user inputs.
- Develop an interactive web application using Flask.
- Deploy the application using Render.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Model | Multiple Linear Regression |
| Backend | Flask |
| Frontend | HTML, CSS |
| Model Serialization | Pickle |
| Deployment | Render |
| Version Control | Git, GitHub |
| Production Server | Gunicorn |

## 📂 Project Structure

```text
House_Price_Prediction/
│
├── dataset/
│   └── data.csv
│
├── static/
│   └── images/
│       └── profile.jpeg
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── house_price_model.pkl
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

## 📊 Dataset Description

The project uses a house sales dataset containing property information and historical selling prices.

**Important Features:**

| Feature | Description |
|---|---|
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft_living | Living area in square feet |
| sqft_lot | Lot area in square feet |
| floors | Number of floors |
| condition | Overall condition of the property |
| yr_built | Year the property was built |
| city | Location of the property |
| statezip | State and ZIP code |
| price | Target variable (house price) |

## ⚙️ Machine Learning Workflow

1. Data Loading
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Feature Engineering
6. Train-Test Split
7. Model Training using Multiple Linear Regression
8. Model Evaluation using R² Score and RMSE
9. Model Serialization using Pickle

The trained model is saved as `house_price_model.pkl` and loaded by the Flask application for predictions.

## 🤖 Machine Learning Algorithm

**Multiple Linear Regression**

Multiple Linear Regression is a supervised machine learning algorithm used to predict a continuous target variable using multiple independent variables.

In this project:

- Independent Variables: Property features such as bedrooms, bathrooms, living area, lot size, floors, condition, year built, city, and ZIP code.
- Dependent Variable: House Price.

The model learns relationships between property features and house prices to generate predictions.

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/sharvanboppa/House_Price_Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd House_Price_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Machine Learning Model (Optional)

If you want to retrain the model using the dataset:

```bash
python train_model.py
```

This generates the trained model file:

```text
house_price_model.pkl
```

### 7. Run the Flask Application

```bash
python app.py
```

### 8. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## 🌐 Web Application Features

- Interactive house price prediction form.
- User-friendly interface.
- Property details input.
- City selection and ZIP code input.
- Machine Learning-based price prediction.
- Flask backend integration.
- Responsive web design.

## ☁️ Deployment on Render

The application can be deployed using Render.

**Deployment Configuration:**

| Setting | Value |
|---|---|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Branch | `main` |

**Deployment Steps:**

1. Push the project code to GitHub.
2. Log in to Render.
3. Create a Web Service or select an existing service.
4. Connect the GitHub repository.
5. Configure the build and start commands.
6. Deploy the application.
7. Open the generated application URL.

**Note:** The `house_price_model.pkl` file must be included in the GitHub repository because the Flask application loads this trained model when it starts.

## 📈 Model Evaluation

The model is evaluated using the following metrics:

- **R² Score:** Measures how well the model explains the variation in house prices.
- **RMSE (Root Mean Squared Error):** Measures the average prediction error, with greater weight given to larger errors.

Actual evaluation results are generated when running `train_model.py`.

## 🔮 Future Enhancements

- Experiment with additional regression algorithms.
- Improve model performance through hyperparameter tuning.
- Add interactive data visualizations.
- Improve prediction accuracy through feature engineering.
- Integrate additional property-related features.

## 👨‍💻 Author

**Sharvan Boppa**

- GitHub: [sharvanboppa](https://github.com/sharvanboppa)

## ⭐ Acknowledgment

This project was developed as a practical implementation of Machine Learning, Data Analysis, and Flask Web Development concepts.

If you find this project useful, consider giving the repository a ⭐.

---

**House Price Prediction | Machine Learning | Python | Flask**
