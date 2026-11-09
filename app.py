import pandas as pd
import pickle
from flask import Flask, render_template, request
app = Flask(__name__)
# Load Dataset
df = pd.read_csv("dataset/data.csv")
cities = sorted(df["city"].dropna().unique())
if "Seattle" in cities:
    cities.remove("Seattle")
    cities.insert(0, "Seattle")
city_mapping = {city: i for i, city in enumerate(cities)}
# Load Machine Learning Model
with open("house_price_model.pkl", "rb") as file:
    model_data = pickle.load(file)
reg = model_data["model"]
model_columns = model_data["columns"]
# Home Page
@app.route("/")
def home():
    return render_template(
        "index.html",
        cities=cities,
        city_mapping=city_mapping,
        selected_city=""
    )
# Predict House Price
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            key: float(request.form[key])
            for key in [
                "bedrooms", "bathrooms", "sqft_living",
                "sqft_lot", "floors", "condition", "yr_built"
            ]
        }

        data["condition"] = int(data["condition"])
        data["yr_built"] = int(data["yr_built"])

        data["city"] = request.form["city"]
        data["statezip"] = request.form["statezip"]

        if data["city"] not in city_mapping:
            raise ValueError("Invalid city")

        X = pd.get_dummies(
            pd.DataFrame([data]), dtype=int
        )

        X = X.reindex(
            columns=model_columns,
            fill_value=0
        )

        price = reg.predict(X)[0]

        return render_template(
            "index.html",
            cities=cities,
            city_mapping=city_mapping,
            selected_city=data["city"],
            prediction_text=f"Estimated House Price: $ {price:,.2f}"
        )

    except (ValueError, KeyError):
        error = "Please enter valid values in all fields."

    except Exception:
        app.logger.exception("Prediction failed")
        error = "Unable to predict house price."

    return render_template(
        "index.html",
        cities=cities,
        city_mapping=city_mapping,
        selected_city=request.form.get("city", ""),
        error_text=error
    )
# Run Application
if __name__ == "__main__":
    app.run(debug=False)
