import os
from src.pipeline import MLClassifierPipeline

if __name__ == '__main__':
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, 'data', 'train.csv')
    test_csv = os.path.join(base_dir, 'data', 'test.csv')
    
    # Initialize and execute pipeline
    pipeline = MLClassifierPipeline(train_path=train_csv, test_path=test_csv)
    pipeline.run()
