import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# CUSTOM TRANSFORMERS & PREPROCESSING FUNCTIONS
# =====================================================================

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean text data before feeding it to TF-IDF.
    Converts to lowercase, removes punctuation, special characters, and double spaces.
    Handles NaN values by converting them to empty strings.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        # Convert series or array to list of strings
        if isinstance(X, pd.DataFrame):
            # If a DataFrame is passed, take the first column
            X_clean = X.iloc[:, 0].fillna("").astype(str)
        else:
            X_clean = pd.Series(X).fillna("").astype(str)
            
        # Clean text
        X_clean = X_clean.apply(self._clean_text)
        return X_clean

    def _clean_text(self, text):
        # Lowercase
        text = text.lower()
        # Remove common platform specific tags
        text = re.sub(r'\(website hidden by airbnb\)', '', text)
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def build_preprocessor(geo_cols, skewed_cols, categorical_cols, text_col):
    """
    Builds a ColumnTransformer that preprocesses geo, skewed numeric, categorical, and text columns.
    
    - Geo: Imputes missing with median -> Standardizes.
    - Skewed: Imputes missing with median -> Log1p transform -> Standardizes.
    - Categorical: Imputes missing with mode -> One-Hot Encodes.
    - Text: Cleans text using TextCleaner -> TF-IDF Vectorization.
    """
    
    # 1. Geo Pipeline (No log transformation!)
    geo_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Skewed Pipeline (Apply log transformation)
    skewed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])
    
    # 3. Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 4. Text Pipeline (using description column)
    text_transformer = Pipeline(steps=[
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)))
    ])
    
    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('geo', geo_transformer, geo_cols),
            ('skewed', skewed_transformer, skewed_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('text', text_transformer, text_col)
        ],
        remainder='drop'  # Drop other columns (like property_id)
    )
    
    return preprocessor


# =====================================================================
# PIPELINE COORDINATOR CLASS
# =====================================================================

class MLClassifierPipeline:
    """
    Coordinates data loading, pipeline building, training, validation,
    evaluation, and generating final predictions on test data.
    """
    def __init__(self, train_path, test_path, target_col='price_tier'):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.model = None
        self.pipeline = None
        
    def run(self):
        print("--- Step 1: Loading Data ---")
        train_df, test_df = self._load_data()
        
        # Define features
        geo_cols = ['latitude', 'longitude']
        skewed_cols = ['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']
        categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
        text_col = 'description'
        
        all_feature_cols = geo_cols + skewed_cols + categorical_cols + [text_col]
        X = train_df[all_feature_cols]
        y = train_df[self.target_col]
        
        print(f"Loaded training data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Target distribution:\n{y.value_counts(normalize=True).round(4) * 100}")
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\n--- Step 2: Building Pipeline ---")
        preprocessor = build_preprocessor(geo_cols, skewed_cols, categorical_cols, text_col)
        
        # Complete pipeline containing preprocessing and the classifier
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1))
        ])
        
        print("\n--- Step 3: Training and Local Validation ---")
        print("Fitting pipeline on training split...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"\nLocal Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred))
        
        print("\n--- Step 4: Retraining on Entire Training Set ---")
        print("Fitting pipeline on all training data...")
        self.pipeline.fit(X, y)
        
        print("\n--- Step 5: Generating Predictions on Test Set ---")
        X_test = test_df[all_feature_cols]
        test_preds = self.pipeline.predict(X_test)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'property_id': test_df['property_id'],
            'predicted_price_tier': test_preds
        })
        
        output_path = os.path.join(os.path.dirname(self.test_path), 'predictions.csv')
        output_df.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")
        print(f"Predictions distribution:\n{output_df['predicted_price_tier'].value_counts()}")
        print("\nPipeline execution complete successfully!")

    def _load_data(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training file not found at: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test file not found at: {self.test_path}")
            
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, test_df


if __name__ == '__main__':
    # Define file paths
    # Using relative paths from project root or absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, 'data', 'train.csv')
    test_csv = os.path.join(base_dir, 'data', 'test.csv')
    
    pipeline = MLClassifierPipeline(train_path=train_csv, test_path=test_csv)
    pipeline.run()
