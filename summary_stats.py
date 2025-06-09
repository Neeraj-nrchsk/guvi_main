import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummaryStats:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
        self.data = data.copy()
        self.numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(exclude=np.number).columns.tolist()
        
    def get_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive statistical summary with enhanced formatting
        Includes: count, mean, std, min, max, percentiles, unique counts, and missing values
        """
        summary = self.data.describe(percentiles=[.05, .25, .5, .75, .95]).T
        summary['dtype'] = self.data.dtypes
        summary['n_unique'] = self.data.nunique()
        summary['missing'] = self.data.isnull().sum()
        summary['variance'] = self.data[self.numeric_cols].var()
        
        # Reorder columns for better readability
        return summary[['dtype', 'count', 'missing', 'mean', 'std', 'variance', 
                       'min', '5%', '25%', '50%', '75%', '95%', 'max', 'n_unique']]

    def get_insights(self) -> Dict[str, Any]:
        """
        Generate domain-specific insights for Iris dataset with biological context
        """
        insights = {
            'biological_validity': self._check_biological_constraints(),
            'species_analysis': self._analyze_species(),
            'feature_variance': self._get_variance_analysis(),
            'correlation_analysis': self._get_correlation_insights(),
            'outlier_analysis': self._get_outlier_analysis()
        }
        return insights

    def _check_biological_constraints(self) -> Dict[str, bool]:
        """Verify biological plausibility of measurements"""
        df = self.data
        valid = {}
        if all(col in df.columns for col in ['SepalLengthCm', 'SepalWidthCm']):
            valid['valid_sepal_ratio'] = ((df['SepalLengthCm'] / df['SepalWidthCm']) < 10).all()
        if all(col in df.columns for col in ['PetalLengthCm', 'PetalWidthCm']):
            valid['valid_petal_ratio'] = ((df['PetalLengthCm'] / df['PetalWidthCm']) < 10).all()
        if 'Species' in df.columns:
            valid_species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            valid['valid_species'] = df['Species'].isin(valid_species).all()
        return valid

    def _analyze_species(self) -> Dict[str, Any]:
        """Analyze species distribution and characteristics"""
        analysis = {}
        if 'Species' in self.data.columns:
            species_counts = self.data['Species'].value_counts()
            analysis.update({
                'most_common': species_counts.idxmax(),
                'least_common': species_counts.idxmin(),
                'distribution': species_counts.to_dict(),
                'measurement_means': self.data.groupby('Species')[self.numeric_cols].mean().to_dict()
            })
        return analysis

    def _get_variance_analysis(self) -> Dict[str, Any]:
        """Analyze feature variability"""
        analysis = {}
        if self.numeric_cols:
            variances = self.data[self.numeric_cols].var()
            analysis.update({
                'highest_variance': variances.idxmax(),
                'lowest_variance': variances.idxmin(),
                'variance_ranking': variances.sort_values(ascending=False).to_dict()
            })
        return analysis

    def _get_correlation_insights(self) -> Dict[str, Any]:
        """Identify significant correlations"""
        insights = {}
        if len(self.numeric_cols) > 1:
            corr_matrix = self.data[self.numeric_cols].corr().abs().round(2)
            insights['top_correlations'] = self._get_top_correlations(corr_matrix)
            insights['correlation_matrix'] = corr_matrix.to_dict()
        return insights

    def _get_top_correlations(self, corr_matrix: pd.DataFrame, n: int = 3) -> Dict[str, float]:
        """Extract top N strongest correlations, robust to dtype issues"""
        # Ensure corr_matrix is float type
        corr_matrix = corr_matrix.astype(float)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        sorted_corr = upper.stack().sort_values(ascending=False)
        return sorted_corr.head(n).to_dict()

    def _get_outlier_analysis(self, iqr_factor: float = 1.5) -> Dict[str, Any]:
        """Identify outliers using IQR method"""
        outliers = {}
        for col in self.numeric_cols:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            if iqr < 1e-6:  # Skip low-variance features
                continue
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            outlier_count = ((self.data[col] < lower) | (self.data[col] > upper)).sum()
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round(outlier_count / len(self.data) * 100, 2),
                'bounds': (round(lower, 2), round(upper, 2))
            }
        return outliers

    def formatted_report(self) -> str:
        """Generate human-readable report"""
        report = []
        report.append("="*40)
        report.append("IRIS DATASET ANALYSIS REPORT".center(40))
        report.append("="*40)
        
        # Summary Statistics
        report.append("\n[SUMMARY STATISTICS]")
        report.append(self.get_summary().to_string())
        
        # Insights
        insights = self.get_insights()
        report.append("\n\n[BIOLOGICAL VALIDATION]")
        for k, v in insights['biological_validity'].items():
            report.append(f"- {k.replace('_', ' ').title()}: {v}")
            
        report.append("\n[SPECIES ANALYSIS]")
        if insights['species_analysis']:
            report.append(f"Most Common Species: {insights['species_analysis']['most_common']}")
            report.append(f"Measurement Means:\n{pd.DataFrame(insights['species_analysis']['measurement_means'])}")
            
        report.append("\n[VARIANCE ANALYSIS]")
        report.append(f"Highest Variance Feature: {insights['feature_variance']['highest_variance']}")
        
        report.append("\n[TOP CORRELATIONS]")
        for pair, value in insights['correlation_analysis']['top_correlations'].items():
            report.append(f"- {pair}: {value}")
            
        report.append("\n[OUTLIER ANALYSIS]")
        for col, stats in insights['outlier_analysis'].items():
            report.append(f"{col}: {stats['count']} outliers ({stats['percentage']}%)")
            
        return "\n".join(report)

# Usage Example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Iris.csv")
    
    # If you have a FeatureEngineering class, import it at the top of the file.
    # Otherwise, define a minimal placeholder here:
    class FeatureEngineering:
        def __init__(self, data):
            self.data = data.copy()
        def engineer_features(self):
            # Placeholder: return self for chaining
            return self
        def scale_features(self):
            # Placeholder: return the original data
            return self.data

    fe = FeatureEngineering(df)
    df_processed = fe.engineer_features().scale_features()
    
    # Generate summary and insights
    stats = SummaryStats(df_processed)
    print(stats.formatted_report())
    print("\nDetailed Insights:")
    print(stats.get_insights())