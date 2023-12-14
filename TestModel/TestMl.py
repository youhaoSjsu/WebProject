import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data as a list of tuples (replace this with your actual data)
from django.db import connection


def execute_custom_sql(query, params=None):
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        result = cursor.fetchall()
    return result


result = execute_custom_sql("SELECT e.name, e.category, e.description, u.age, u.job, "
                            "GROUP_CONCAT(h.name ORDER BY h.name) AS hobbies, p.rate,p.rate_id "
                            "FROM testmodel_event e JOIN testmodel_pushrate p ON e.event_id = p.event_id "
                            "JOIN testmodel_userprofile u ON u.user_id = p.user_id "
                            "JOIN testmodel_userhobby uh ON p.user_id = uh.user_id "
                            "JOIN testmodel_hobby h ON uh.hobby_id = h.hobby_id "
                            "group by p.rate_id order by p.rate_id asc;")
data = result
# data = [
#     ("Event1", "Category1", "Description1", "Hobby1", 25, "Job1", 4.5),
#     ("Event2", "Category2", "Description2", "Hobby2", 30, "Job2", 3.0),
#     # Add more tuples...
# ]

# Convert the list of tuples to a DataFrame
columns = ["event_name", "event_category", "event_description", "user_hobby", "user_age", "user_job", "rating_score"]
df = pd.DataFrame(data, columns=columns)

# Feature Engineering
# Encode categorical features
label_encoder = LabelEncoder()
df['user_hobby'] = label_encoder.fit_transform(df['user_hobby'])
df['user_job'] = label_encoder.fit_transform(df['user_job'])
df['event_category'] = label_encoder.fit_transform(df['event_category'])

# Combine text features into one
df['event_text'] = df['event_name'] + ' ' + df['event_description']
df['user_profile'] = df['user_hobby'].astype(str) + ' ' + df['user_age'].astype(str) + ' ' + df['user_job'].astype(str)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# TF-IDF Vectorization for text features
tfidf_vectorizer_event = TfidfVectorizer(stop_words='english')
tfidf_vectorizer_user = TfidfVectorizer(stop_words='english')

event_text_matrix = tfidf_vectorizer_event.fit_transform(train_data['event_text'])
user_profile_matrix = tfidf_vectorizer_user.fit_transform(train_data['user_profile'])

# Combine the TF-IDF matrices with numerical features
X_train = pd.concat([train_data[['user_hobby', 'user_age', 'user_job', 'event_category']],
                     pd.DataFrame(event_text_matrix.toarray(), columns=tfidf_vectorizer_event.get_feature_names_out()),
                     pd.DataFrame(user_profile_matrix.toarray(),
                                  columns=tfidf_vectorizer_user.get_feature_names_out())],
                    axis=1)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, train_data['rating_score'])

# Make predictions on the test set
event_text_matrix_test = tfidf_vectorizer_event.transform(test_data['event_text'])
user_profile_matrix_test = tfidf_vectorizer_user.transform(test_data['user_profile'])

X_test = pd.concat([test_data[['user_hobby', 'user_age', 'user_job', 'event_category']],
                    pd.DataFrame(event_text_matrix_test.toarray(),
                                 columns=tfidf_vectorizer_event.get_feature_names_out()),
                    pd.DataFrame(user_profile_matrix_test.toarray(),
                                 columns=tfidf_vectorizer_user.get_feature_names_out())],
                   axis=1)

predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(test_data['rating_score'], predictions)
print(f'Mean Squared Error: {mse}')
