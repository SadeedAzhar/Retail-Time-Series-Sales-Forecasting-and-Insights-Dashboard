import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("train.csv")


df['Year'] = pd.to_datetime(df['date']).dt.year
df['Month'] = pd.to_datetime(df['date']).dt.month
df['Day'] = pd.to_datetime(df['date']).dt.day
df['Weekday'] = pd.to_datetime(df['date']).dt.weekday

# Drop unnecessary columns
df = df.drop(columns=['date', 'id'], errors='ignore') 

# Separate features and target
X = df.drop(columns='sales')
y = df['sales']

# Handle 'family' label encoding separately
le = LabelEncoder()
X['family'] = le.fit_transform(X['family'])

# Save the label encoder
joblib.dump(le, "label_encoder.pkl")

# Defining categorical and numerical columns
cat_cols = [col for col in X.select_dtypes(include='object').columns if col != 'family']  
num_cols = [col for col in X.columns if col not in cat_cols]

# Defining preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Save preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")

# Split and train the model
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42,max_depth=None,min_samples_leaf=6,min_samples_split=20)
model.fit(X_train, y_train)

# Saving the model
joblib.dump(model, "model.pkl")

