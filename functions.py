import pandas as pd

# Load data from a given URL and preprocess column names
def load_data(url):
    db = pd.read_csv(url)
    db.rename(columns={"ST": "State"}, inplace=True)
    db.columns = db.columns.str.lower()
    db.columns = db.columns.str.replace(" ", "_")
    return db

# Clean the 'gender' column by mapping various representations to 'F' and 'M'
def clean_gender(db):
    gender_map = {
        'F': 'F',
        'Femal': 'F',
        'female': 'F',
        'M': 'M',
        'Male': 'M'
    }
    db['gender'] = db['gender'].replace(gender_map)
    return db

# Clean the 'state' column by mapping various representations to standardized state names
def clean_state(db):
    state_map = {
        'Washington': 'Washington',
        'WA': 'Washington',
        'Arizona': 'Arizona',
        'AZ': 'Arizona',
        'California': 'California',
        'Cali': 'California',
    }
    db['state'] = db['state'].replace(state_map)
    return db

# Clean the 'education' column by mapping various representations to standardized education levels
def clean_education(db):
    education_map = {
        'High School or Below': 'High School or Below',
        'Bachelors': 'Bachelor',
        'Master': 'Master',
        'Doctor': 'Doctor',
        'College ': 'College',
        ' Bachelor': 'Bachelor',
    }
    db['education'] = db['education'].replace(education_map)
    return db

# Clean the 'customer_lifetime_value' column by removing '%' and converting to float
def clean_customer_lifetime_value(db):
    db['customer_lifetime_value'] = db['customer_lifetime_value'].str.rstrip('%').astype(float)
    return db

# Helper function to extract the number of open complaints from a string
def extract_complaints(value):
    try:
        return int(value.split('/')[1])
    except (IndexError, ValueError):
        return 0

# Clean the 'number_of_open_complaints' column by extracting the number of complaints
def clean_number_of_open_complaints(db):
    db['number_of_open_complaints'] = db['number_of_open_complaints'].astype(str)
    db['number_of_open_complaints'] = db['number_of_open_complaints'].apply(extract_complaints)
    return db

# Clean the 'vehicle_class' column by mapping various representations to 'Luxury'
def clean_vehicle_class(db):
    vehicle_class_map = {
        'Sports Car': 'Luxury',
        'Luxury SUV': 'Luxury',
        'Luxury Car': 'Luxury'
    }
    db['vehicle_class'] = db['vehicle_class'].replace(vehicle_class_map)
    return db

# Handle null values by filling numerical columns with the mean and categorical columns with the mode
def handle_null_values(db):
    numerical_columns = db.select_dtypes(include=['float64', 'int64']).columns
    db[numerical_columns] = db[numerical_columns].fillna(db[numerical_columns].mean())
    categorical_columns = db.select_dtypes(include=['object']).columns
    db[categorical_columns] = db[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))
    return db

# Handle duplicate rows by dropping them and resetting the index
def handle_duplicates(db):
    db = db.drop_duplicates()
    db = db.reset_index(drop=True)
    return db

# Save the cleaned dataset to a CSV file
def save_cleaned_data(db, filename='cleaned_data.csv'):
    db.to_csv(filename, index=False)

# Review statistics for total claim amount and customer lifetime value
def review_statistics(db):
    stats = db[['total_claim_amount', 'customer_lifetime_value']].describe()
    return stats

# Identify customers with high policy claim amount and low customer lifetime value
def identify_customers(db):
    high_claim_threshold = db['total_claim_amount'].quantile(0.75)
    low_clv_threshold = db['customer_lifetime_value'].quantile(0.25)
    filtered_db = db[(db['total_claim_amount'] > high_claim_threshold) & (db['customer_lifetime_value'] < low_clv_threshold)]
    return filtered_db

# Calculate summary statistics for high policy claim amount and low customer lifetime value
def calculate_summary_statistics(filtered_db):
    summary_stats = filtered_db[['total_claim_amount', 'customer_lifetime_value']].describe()
    return summary_stats