
import pandas as pd
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "data.csv"

MODEL_PATH = BASE_DIR / "house_price_model.pkl"


# ==========================================
# HOUSE PRICE PREDICTION MODEL
# ==========================================

class HousePriceModel:

    def __init__(self, file_path):

        self.file_path = file_path

        self.model = LinearRegression()

        self.feature_columns = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "condition",
            "yr_built",
            "city",
            "statezip"
        ]


    # ======================================
    # 1. LOAD DATASET
    # ======================================

    def load_data(self):

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        self.df = pd.read_csv(self.file_path)

        print("\nDataset Loaded Successfully!")

        print("\nDataset Shape:", self.df.shape)

        print("\nFirst Five Records:")

        print(self.df.head())


    # ======================================
    # 2. CLEAN DATASET
    # ======================================

    def clean_data(self):

        print("\nStarting Data Cleaning...")

        # Remove unnecessary columns if available

        columns_to_drop = [
            "date",
            "street"
        ]

        self.df.drop(
            columns=columns_to_drop,
            errors="ignore",
            inplace=True
        )

        # Check required columns

        required_columns = self.feature_columns + ["price"]

        missing_columns = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing dataset columns: "
                + ", ".join(missing_columns)
            )

        # Clean city names

        self.df["city"] = (
            self.df["city"]
            .astype("string")
            .str.strip()
        )

        # Remove empty city names

        self.df["city"] = self.df["city"].replace(
            "", pd.NA
        )

        # Extract five-digit ZIP codes

        self.df["statezip"] = (
            self.df["statezip"]
            .astype(str)
            .str.extract(r"(\d{5})", expand=False)
        )

        # Convert numeric columns

        numeric_columns = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "condition",
            "yr_built",
            "statezip",
            "price"
        ]

        for col in numeric_columns:

            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

        # Remove infinite values

        self.df.replace(
            [float("inf"), float("-inf")],
            pd.NA,
            inplace=True
        )

        # Remove missing values

        self.df.dropna(
            subset=required_columns,
            inplace=True
        )

        # Keep only positive house prices

        self.df = self.df[
            self.df["price"] > 0
        ].copy()

        # Validate house condition

        self.df = self.df[
            self.df["condition"].between(1, 5)
        ].copy()

        # Keep valid ZIP codes

        self.df = self.df[
            self.df["statezip"].between(10000, 99999)
        ].copy()

        if self.df.empty:

            raise ValueError(
                "No valid records available after data cleaning."
            )

        print("\nData Cleaning Completed!")

        print("Cleaned Dataset Shape:", self.df.shape)

        print("\nMissing Values:")

        print(self.df[required_columns].isnull().sum())


    # ======================================
    # 3. CREATE CITY MAPPING
    # ======================================

    def create_mappings(self):

        self.cities = sorted(
            self.df["city"].unique().tolist()
        )

        # Keep Seattle as the first city

        if "Seattle" in self.cities:

            self.cities.remove("Seattle")

            self.cities.insert(0, "Seattle")

        # Create numerical city mapping

        self.city_mapping = {

            city: index

            for index, city in enumerate(self.cities)

        }

        print("\nCity Mapping Created Successfully!")

        print("\nTotal Cities:", len(self.cities))

        print("\nCity Mapping:")

        print(self.city_mapping)


    # ======================================
    # 4. PREPARE TRAINING DATA
    # ======================================

    def prepare_data(self):

        self.X = self.df[
            self.feature_columns
        ].copy()

        self.y = self.df["price"].copy()

        # Convert city names into numerical values

        self.X["city"] = self.X["city"].map(
            self.city_mapping
        )

        # Ensure numerical feature values

        self.X = self.X.astype(float)

        print("\nTraining Data Prepared Successfully!")

        print("\nFeature Columns:")

        print(self.X.columns.tolist())

        print("\nInput Shape:", self.X.shape)

        print("\nOutput Shape:", self.y.shape)


    # ======================================
    # 5. TRAIN AND EVALUATE MODEL
    # ======================================

    def train_model(self):

        print("\nStarting Linear Regression Training...")

        # Split dataset

        X_train, X_test, y_train, y_test = train_test_split(

            self.X,

            self.y,

            test_size=0.2,

            random_state=42

        )

        # Train model on training data

        self.model.fit(
            X_train,
            y_train
        )

        # Predictions

        train_prediction = self.model.predict(
            X_train
        )

        test_prediction = self.model.predict(
            X_test
        )

        # Training evaluation

        train_r2 = r2_score(
            y_train,
            train_prediction
        )

        train_rmse = mean_squared_error(
            y_train,
            train_prediction
        ) ** 0.5

        # Testing evaluation

        test_r2 = r2_score(
            y_test,
            test_prediction
        )

        test_rmse = mean_squared_error(
            y_test,
            test_prediction
        ) ** 0.5

        print("\n========== MODEL EVALUATION ==========")

        print("\nTraining R2 Score:", round(train_r2, 4))

        print("Training RMSE:", round(train_rmse, 2))

        print("\nTesting R2 Score:", round(test_r2, 4))

        print("Testing RMSE:", round(test_rmse, 2))

        print("\n======================================")

        # Retrain using the complete dataset
        # for the final deployed model

        self.model.fit(
            self.X,
            self.y
        )

        print(
            "\nFinal Model Trained Using Complete Dataset!"
        )


    # ======================================
    # 6. SAVE TRAINED MODEL
    # ======================================

    def save_model(self):

        model_data = {

            "model": self.model,

            "columns": self.feature_columns,

            "cities": self.cities,

            "city_mapping": self.city_mapping

        }

        with open(MODEL_PATH, "wb") as file:

            pickle.dump(
                model_data,
                file
            )

        print("\nModel Saved Successfully!")

        print("\nModel Location:")

        print(MODEL_PATH)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    obj = HousePriceModel(DATA_PATH)

    obj.load_data()

    obj.clean_data()

    obj.create_mappings()

    obj.prepare_data()

    obj.train_model()

    obj.save_model()

    print("\nHOUSE PRICE PREDICTION MODEL COMPLETED!")
