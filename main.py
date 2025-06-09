from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from data_integrity import DataIntegrity
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandler
from visualization import Visualization

if __name__ == "__main__":
    # Step 1: Load and clean data using local CSV
    loader = DataLoader("Iris.csv")
    data = loader.load_data()
    print("Data summary after load:", loader.data_summary)

    # Step 2: Feature selection and engineering
    fe = FeatureEngineering(data)
    data = fe.select_features(exclude=['Id'])
    data = fe.engineer_features()
    print("\nEngineered Features:")
    print(fe.get_feature_descriptions())
    print("\nSample Data:")
    print(data[['sepal_ratio', 'petal_area', 'Species_encoded']].head())

    # Step 2b: Feature scaling (standardization)
    data = fe.scale_features(method='standard')
    print("\nScaled Features Summary:")
    # Only print columns that exist after scaling
    scaled_cols = [col for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'sepal_ratio', 'petal_ratio'] if col in data.columns]
    print(data[scaled_cols].describe().round(2))
    print("\nSample Data After Scaling:")
    sample_cols = [col for col in ['sepal_ratio', 'petal_area', 'Species_encoded'] if col in data.columns]
    print(data[sample_cols].head())

    # Step 3: Data integrity and consistency
    integrity = DataIntegrity(data)
    integrity_report = integrity.check_integrity(verbose=True)
    data = integrity.ensure_consistency()
    print("Data summary after integrity checks:", integrity.data.shape)

    # Step 4: Outlier detection and handling
    outlier_handler = OutlierHandler(data)
    outlier_report = outlier_handler.detect(method='iqr', group_by='Species')
    print("\nOutlier Report (by Species, IQR):", outlier_report)
    # Remove/cap outliers (optional, can use 'remove', 'cap', or 'impute')
    data = outlier_handler.handle(strategy='cap', method='iqr')
    print("Data shape after outlier handling:", data.shape)

    # Step 5: Summary statistics and insights
    stats = SummaryStats(data)
    print(stats.get_summary())
    print(stats.get_insights())

    # Step 6: Pattern analysis
    patterns = PatternAnalysis(data)
    patterns.print_patterns()

    # Step 7: Visualization
    viz = Visualization(data)
    viz.plot_key_findings()