import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from django.db import connection
from django.core.wsgi import get_wsgi_application
import os
from TestModel.models import *

# Initialize Django application
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WebProject.settings")
application = get_wsgi_application()


def execute_custom_sql(query, params=None):
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        result = cursor.fetchall()
    return result


def load_data():
    result = execute_custom_sql("SELECT e.name, e.category, e.description, u.age, u.job, "
                                "GROUP_CONCAT(h.name ORDER BY h.name) AS hobbies, p.rate "
                                "FROM testmodel_event e JOIN testmodel_pushrate p ON e.event_id = p.event_id "
                                "JOIN testmodel_userprofile u ON u.user_id = p.user_id "
                                "JOIN testmodel_userhobby uh ON p.user_id = uh.user_id "
                                "JOIN testmodel_hobby h ON uh.hobby_id = h.hobby_id "
                                "GROUP BY p.rate_id ORDER BY p.rate_id ASC;")
    columns = ["event_name", "event_category", "event_description", "user_hobby", "user_age", "user_job",
               "rating_score"]
    return pd.DataFrame(result, columns=columns)


def preprocess_data(df):
    # Initialize label encoders
    le_hobby = LabelEncoder()
    le_job = LabelEncoder()
    le_category = LabelEncoder()

    # Combine text features into one
    df['event_text'] = df['event_name'] + ' ' + df['event_category'] + '' + df['event_description']
    df['user_profile'] = df['user_hobby'].astype(str) + ' ' + df['user_age'].astype(str) + ' ' + df['user_job'].astype(
        str)

    # Encode categorical features
    df['user_hobby_encoded'] = le_hobby.fit_transform(df['user_hobby'])
    df['user_job_encoded'] = le_job.fit_transform(df['user_job'])
    df['event_category_encoded'] = le_category.fit_transform(df['event_category'])

    return df


def preprocess_data(df):
    label_encoder = LabelEncoder()

    # Encoding all string columns except 'user_age' and 'rating_score'
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])

    # Standardizing numerical columns (if needed)
    scaler = StandardScaler()
    df['user_age'] = scaler.fit_transform(df[['user_age']])

    return df


def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    # Load data
    df = load_data()

    # Preprocess data
    df_processed = preprocess_data(df)

    # Split the data
    X = df_processed.drop('rating_score', axis=1)
    y = df_processed['rating_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    joblib.dump(model, 'C:\sjsu\CS156\WebProject\WebProject\mlModel.pkl')


if __name__ == "__main__":
    main()
