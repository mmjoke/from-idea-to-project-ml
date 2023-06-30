import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from pickle import dump, load

def open_data(path="D:/ml/ml-learn/ml-seminars/hse-ml/from_idea_to_prot_ML/data/cars.csv"):
    df = pd.read_csv(path)
    return df


def get_X_y_data(df: pd.DataFrame):
    y = df["selling_price"]
    X = df[["name", "year", "km_driven", "fuel", "seller_type", "transmission",
            "owner", "mileage", "engine", "max_power", "torque", "seats"]]
    return X, y


def preprocess_data(df: pd.DataFrame, test=True):
    df = df.dropna(axis="index")

    if test:
        X_df, y_df = get_X_y_data(df)

    else:
        X_df = df
    
    # Исключаем столбец torgue из выборки, так как содежит текстовые данные, вызывающие трудности при обработке
    # Исключим также столбец name
    X_df = X_df.drop(axis=1, labels=["torque", "name"])
    
    for feature in ["mileage", "engine", "max_power"]:
        X_df[feature] = X_df[feature].apply(lambda x: x.split()[0]).astype("float64")
    
    X_df["age_less_10"] = X_df["year"] > 2004
    X_df["age_less_10"] = X_df["age_less_10"].astype(dtype="int64")
    
    categorical = ["fuel", "seller_type", "transmission", "owner", "seats", "age_less_10"]
    numeric_features = ["year", "km_driven", "mileage", "engine", "max_power"]

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ('scaling', StandardScaler(), numeric_features)
    ])
    
    X_df = column_transformer.fit_transform(X_df)
    
    new_column_names = list(column_transformer.transformers_[0][1].get_feature_names_out())
    new_column_names.extend(numeric_features)

    X_df = pd.DataFrame(X_df, columns=new_column_names)

    if test:
        return X_df, y_df
    else:
        return X_df

def fit_and_save_model(X_df, y_df, test_size=0.3, random_state=123456789, path="D:/ml/ml-learn/ml-seminars/hse-ml/from_idea_to_prot_ML/data/model_weights.mw"):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_size, random_state=random_state)
    model = Ridge()
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    rmse = mean_squared_error(test_prediction, y_test, squared=False)
    print(f"Model RMSE is {rmse}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="D:/ml/ml-learn/ml-seminars/hse-ml/from_idea_to_prot_ML/data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)
    
    return prediction


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)