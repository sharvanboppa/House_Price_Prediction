
import pandas as pd
import pickle
import logging

from pathlib import Path
from flask import Flask, render_template, request


# ----------------------------------------
# FLASK APPLICATION
# ----------------------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


# ----------------------------------------
# LOAD TRAINED MACHINE LEARNING MODEL
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_price_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    model = model_data["model"]
    columns = model_data["columns"]
    cities = model_data["cities"]
    city_mapping = model_data["city_mapping"]

    app.logger.info("House Price Prediction model loaded successfully.")

except (FileNotFoundError, pickle.UnpicklingError, KeyError) as e:
    raise RuntimeError(
        f"Unable to load the trained model: {e}"
    ) from e


# ----------------------------------------
# DISPLAY WEBSITE
# ----------------------------------------

def display_page(
    selected_city="",
    selected_zip="",
    values=None,
    prediction=None,
    error=None
):

    return render_template(
        "index.html",

        cities=cities,

        city_mapping=city_mapping,

        selected_city=selected_city,

        selected_zip=selected_zip,

        values=values or {},

        prediction=prediction,

        error=error
    )


# ----------------------------------------
# HOME PAGE
# ----------------------------------------

@app.route("/")
def home():

    return display_page()


# ----------------------------------------
# HOUSE PRICE PREDICTION
# ----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get city and ZIP code

    city = request.form.get(
        "city", ""
    ).strip()

    zip_number = request.form.get(
        "statezip", ""
    ).strip()


    # Get all property details

    field_names = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "condition",
        "yr_built"
    ]

    values = {
        key: request.form.get(
            key, ""
        ).strip()

        for key in field_names
    }


    try:

        # --------------------------------
        # 1. VALIDATE CITY
        # --------------------------------

        if city not in city_mapping:

            raise ValueError(
                "Please select a valid city."
            )


        # --------------------------------
        # 2. VALIDATE ZIP CODE
        # --------------------------------

        if not zip_number.isdigit() or len(zip_number) != 5:

            raise ValueError(
                "Please enter a valid 5-digit ZIP code."
            )


        # --------------------------------
        # 3. CONVERT INPUT VALUES
        # --------------------------------

        data = {}

        # Integer fields

        integer_fields = [
            "bedrooms",
            "sqft_living",
            "sqft_lot",
            "condition",
            "yr_built"
        ]

        for key in integer_fields:

            value = values[key]

            if not value.isdigit():

                raise ValueError(
                    f"Please enter a valid whole number for {key.replace('_', ' ')}."
                )

            data[key] = int(value)


        # Decimal fields

        decimal_fields = [
            "bathrooms",
            "floors"
        ]

        for key in decimal_fields:

            try:

                data[key] = float(values[key])

            except (ValueError, TypeError):

                raise ValueError(
                    f"Please enter a valid number for {key}."
                )

            if not pd.notna(data[key]) or data[key] == float("inf") or data[key] == float("-inf"):

                raise ValueError(
                    f"Please enter a valid number for {key}."
                )


        # --------------------------------
        # 4. VALIDATE POSITIVE VALUES
        # --------------------------------

        if any(value <= 0 for value in data.values()):

            raise ValueError(
                "Please enter positive property values."
            )


        # --------------------------------
        # 5. VALIDATE HOUSE CONDITION
        # --------------------------------

        if not 1 <= data["condition"] <= 5:

            raise ValueError(
                "House condition must be between 1 and 5."
            )


        # --------------------------------
        # 6. VALIDATE CONSTRUCTION YEAR
        # --------------------------------

        if not 1800 <= data["yr_built"] <= 2026:

            raise ValueError(
                "Please enter a valid construction year."
            )


        # --------------------------------
        # 7. CITY NUMERICAL MAPPING
        # --------------------------------

        data["city"] = city_mapping[city]


        # --------------------------------
        # 8. ZIP CODE CONVERSION
        # --------------------------------

        data["statezip"] = int(zip_number)


        # --------------------------------
        # 9. PREPARE MODEL INPUT
        # --------------------------------

        input_data = pd.DataFrame([data])


        # Maintain the original model column order

        missing_columns = [
            col for col in columns
            if col not in input_data.columns
        ]

        if missing_columns:

            raise ValueError(
                "Model input features are missing: "
                + ", ".join(missing_columns)
            )


        input_data = input_data.reindex(
            columns=columns
        )


        # --------------------------------
        # 10. PREDICT HOUSE PRICE
        # --------------------------------

        result = model.predict(input_data)[0]


        # --------------------------------
        # 11. DISPLAY PREDICTION RESULT
        # --------------------------------

        return display_page(

            selected_city=city,

            selected_zip=zip_number,

            values=values,

            prediction=round(float(result), 2)

        )


    # --------------------------------
    # ERROR HANDLING
    # --------------------------------

    except ValueError as e:

        return display_page(

            selected_city=city,

            selected_zip=zip_number,

            values=values,

            error=str(e)

        )


    except Exception:

        app.logger.exception(
            "Error occurred during house price prediction."
        )

        return display_page(

            selected_city=city,

            selected_zip=zip_number,

            values=values,

            error=(
                "Something went wrong while predicting "
                "the house price. Please check your inputs "
                "and try again."
            )

        )


# ----------------------------------------
# RUN FLASK APPLICATION
# ----------------------------------------

if __name__ == "__main__":

    app.run(debug=True)
