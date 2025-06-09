import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

# Configure logging with proper format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternAnalysis")


@dataclass
class AnomalyResult:
    """Data class to store anomaly detection results."""
    outlier_count: int
    outlier_indices: List[int]
    outlier_values: Dict[int, float]
    bounds: Tuple[float, float]
    method: str


class PatternAnalysis:
    """
    Enhanced class for identifying patterns, trends, and anomalies in datasets.
    Provides comprehensive data analysis capabilities with proper error handling.
    """
    
    def __init__(self, data: pd.DataFrame, verbose: bool = False):
        """
        Initialize the PatternAnalysis instance.
        
        Args:
            data: Input DataFrame for analysis
            verbose: Whether to print detailed logs during processing
        
        Raises:
            ValueError: If input is not a DataFrame or is empty
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Input DataFrame is empty")
            
        self.data = data.copy()
        self.verbose = verbose
        self._log_level = logging.INFO if verbose else logging.WARNING
        logger.setLevel(self._log_level)
        
        # Identify column types
        self.numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(exclude=np.number).columns.tolist()
        self.datetime_cols = []
        
        # Try to convert string columns to datetime if they appear to be dates
        for col in self.categorical_cols[:]:  # Use slice copy to avoid modification during iteration
            try:
                pd.to_datetime(self.data[col], errors='raise')
                self.data[col] = pd.to_datetime(self.data[col])
                self.datetime_cols.append(col)
                self.categorical_cols.remove(col)
                logger.info(f"Converted column '{col}' to datetime")
            except (ValueError, TypeError):
                pass
        
        logger.info(f"Initialized with {len(self.numeric_cols)} numeric columns, "
                   f"{len(self.categorical_cols)} categorical columns, and "
                   f"{len(self.datetime_cols)} datetime columns")
    
    def identify_patterns(self) -> Dict[str, Any]:
        """
        Identify key patterns and trends in the dataset.
        
        Returns:
            Dictionary of findings including distributions, correlations, and group statistics
        """
        logger.info("Identifying patterns in data...")
        patterns = {}
        
        # Basic summary statistics
        patterns['basic_stats'] = {
            col: self.data[col].describe().to_dict() 
            for col in self.numeric_cols
        }
        
        # Skewness and kurtosis for distributions
        patterns['distribution_shape'] = {
            col: {
                'skewness': float(stats.skew(self.data[col].dropna())),
                'kurtosis': float(stats.kurtosis(self.data[col].dropna())),
                'normality_test': stats.shapiro(self.data[col].dropna()[:5000])  # Limit sample size
            } for col in self.numeric_cols if self.data[col].dropna().shape[0] > 8
        }
        
        # Group statistics by categorical variables
        patterns['group_statistics'] = {}
        for cat_col in self.categorical_cols:
            if self.data[cat_col].nunique() < 30:  # Skip high cardinality categoricals
                try:
                    grouped_means = self.data.groupby(cat_col)[self.numeric_cols].mean().to_dict()
                    grouped_medians = self.data.groupby(cat_col)[self.numeric_cols].median().to_dict()
                    grouped_counts = self.data.groupby(cat_col).size().to_dict()
                    grouped_std = self.data.groupby(cat_col)[self.numeric_cols].std().to_dict()
                    
                    patterns['group_statistics'][cat_col] = {
                        'means': grouped_means,
                        'medians': grouped_medians,
                        'counts': grouped_counts,
                        'std_devs': grouped_std
                    }
                except Exception as e:
                    logger.warning(f"Error calculating group statistics for {cat_col}: {str(e)}")
        
        # Correlation analysis
        if len(self.numeric_cols) > 1:
            try:
                corr_matrix = self.data[self.numeric_cols].corr().round(3)
                patterns['correlation_matrix'] = corr_matrix.to_dict()
                
                # Identify the highest correlations
                corr_pairs = []
                for i, col1 in enumerate(self.numeric_cols):
                    for col2 in self.numeric_cols[i+1:]:
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:  # Strong correlation threshold
                            corr_pairs.append((col1, col2, corr_val))
                
                patterns['strong_correlations'] = sorted(
                    corr_pairs, 
                    key=lambda x: abs(x[2]), 
                    reverse=True
                )
            except Exception as e:
                logger.warning(f"Error calculating correlations: {str(e)}")
        
        # Time series patterns if datetime columns exist
        if self.datetime_cols:
            patterns['time_patterns'] = self._analyze_time_patterns()
        
        # Missing value patterns
        missing_stats = self.data.isnull().sum().to_dict()
        missing_pcts = (self.data.isnull().mean() * 100).round(2).to_dict()
        patterns['missing_values'] = {
            'counts': missing_stats,
            'percentages': missing_pcts,
            'total_missing_rows': self.data.isnull().any(axis=1).sum()
        }
        
        # Column cardinality for categorical variables
        patterns['categorical_cardinality'] = {
            col: {
                'unique_count': self.data[col].nunique(),
                'top_values': self.data[col].value_counts().head(5).to_dict()
            } for col in self.categorical_cols
        }
        
        return patterns
    
    def find_anomalies(
        self, 
        methods: List[str] = ['iqr', 'zscore', 'isolation_forest'],
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0
    ) -> Dict[str, Dict[str, AnomalyResult]]:
        """
        Identify anomalies using multiple detection methods.
        
        Args:
            methods: List of anomaly detection methods to use
            iqr_factor: Factor for IQR method
            zscore_threshold: Threshold for Z-score method
            
        Returns:
            Dictionary of anomaly results by method and column
        """
        logger.info(f"Finding anomalies using methods: {methods}")
        results = {}
        
        valid_methods = ['iqr', 'zscore', 'modified_zscore', 'isolation_forest']
        methods = [m for m in methods if m in valid_methods]
        
        if not methods:
            logger.warning("No valid anomaly detection methods specified")
            return {}
        
        # IQR method
        if 'iqr' in methods:
            results['iqr'] = self._find_iqr_anomalies(iqr_factor)
            
        # Z-score method
        if 'zscore' in methods:
            results['zscore'] = self._find_zscore_anomalies(zscore_threshold)
            
        # Modified Z-score method (more robust to outliers)
        if 'modified_zscore' in methods:
            results['modified_zscore'] = self._find_modified_zscore_anomalies(threshold=3.5)
        
        # Use Isolation Forest for multivariate anomalies if we have enough numeric columns
        if 'isolation_forest' in methods and len(self.numeric_cols) >= 2:
            try:
                from sklearn.ensemble import IsolationForest
                iso_result = self._find_isolation_forest_anomalies()
                # Always include a key for reporting, even if None
                if iso_result and 'multidimensional' in iso_result:
                    results['isolation_forest'] = iso_result
                else:
                    results['isolation_forest'] = {'multidimensional': None}
            except ImportError:
                logger.warning("sklearn not available, skipping isolation forest analysis")
                results['isolation_forest'] = {'multidimensional': None}
        elif 'isolation_forest' in methods:
            # Not enough numeric columns
            results['isolation_forest'] = {'multidimensional': None}
        
        return results
    
    def cluster_analysis(self, n_clusters: Optional[int] = None, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Perform cluster analysis on numeric data to find natural groupings.
        
        Args:
            n_clusters: Number of clusters to use (if None, will try to determine optimal)
            max_clusters: Maximum number of clusters to try when finding optimal
            
        Returns:
            Dictionary with clustering results
        """
        if len(self.numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for clustering")
            return {}
        
        try:
            # Extract and standardize numeric data
            numeric_data = self.data[self.numeric_cols].copy()
            numeric_data = numeric_data.dropna()
            
            if numeric_data.empty:
                logger.warning("No complete rows available for clustering")
                return {}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                n_clusters = self._find_optimal_clusters(scaled_data, max_clusters)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels back to the original data
            numeric_data['cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = numeric_data[numeric_data['cluster'] == cluster_id]
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(numeric_data) * 100, 2),
                    'means': cluster_data[self.numeric_cols].mean().to_dict(),
                    'std_devs': cluster_data[self.numeric_cols].std().to_dict()
                }
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(scaled_data, clusters)
            except ImportError:
                silhouette = None
                
            return {
                'n_clusters': n_clusters,
                'cluster_centers': {
                    f'center_{i}': {col: val for col, val in zip(self.numeric_cols, center)}
                    for i, center in enumerate(kmeans.cluster_centers_)
                },
                'cluster_stats': cluster_stats,
                'silhouette_score': silhouette,
                'feature_importance': self._calculate_cluster_feature_importance(kmeans, self.numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"Error in cluster analysis: {str(e)}")
            return {'error': str(e)}
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance using variance and correlation.
        
        Returns:
            Dictionary of feature importance scores
        """
        if len(self.numeric_cols) < 2:
            return {self.numeric_cols[0]: 1.0} if self.numeric_cols else {}
        
        try:
            # Calculate coefficient of variation (normalized variance)
            cv_scores = {}
            for col in self.numeric_cols:
                mean = self.data[col].mean()
                if abs(mean) > 1e-10:  # Avoid division by very small values
                    cv = self.data[col].std() / abs(mean)
                    cv_scores[col] = abs(cv)
            
            # Correlation-based importance
            corr_matrix = self.data[self.numeric_cols].corr().abs()
            mean_corr = corr_matrix.sum() / (len(self.numeric_cols) - 1)  # Average correlation with other features
            
            # Combine the metrics (higher variance and correlation = more important)
            importance = {}
            for col in self.numeric_cols:
                # Normalize CV scores to [0,1]
                cv_max = max(cv_scores.values()) if cv_scores else 1
                norm_cv = cv_scores.get(col, 0) / cv_max if cv_max > 0 else 0
                
                # Combine with correlation (weight variance more heavily)
                importance[col] = 0.7 * norm_cv + 0.3 * mean_corr.get(col, 0)
            
            # Normalize to sum to 1
            total = sum(importance.values())
            return {k: round(v/total, 4) if total > 0 else 0 for k, v in importance.items()}
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def generate_visualizations(self, output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate common visualizations for the dataset.
        
        Args:
            output_dir: Directory to save visualizations (if None, won't save)
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        try:
            # Distribution plots for numeric columns
            if self.numeric_cols:
                fig, axes = plt.subplots(len(self.numeric_cols), 1, figsize=(10, 3*len(self.numeric_cols)))
                if len(self.numeric_cols) == 1:
                    axes = [axes]  # Make iterable for single subplot
                
                for i, col in enumerate(self.numeric_cols):
                    sns.histplot(self.data[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                
                plt.tight_layout()
                figures['distributions'] = fig
                
                if output_dir:
                    fig.savefig(f"{output_dir}/distributions.png")
            
            # Correlation heatmap
            if len(self.numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = self.data[self.numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Matrix')
                figures['correlation'] = fig
                
                if output_dir:
                    fig.savefig(f"{output_dir}/correlation.png")
            
            # Boxplots for detecting outliers
            if self.numeric_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                self.data[self.numeric_cols].boxplot(ax=ax)
                ax.set_title('Boxplots for Numeric Features')
                plt.xticks(rotation=90)
                figures['boxplots'] = fig
                
                if output_dir:
                    fig.savefig(f"{output_dir}/boxplots.png")
            
            # PCA visualization if we have enough numeric columns
            if len(self.numeric_cols) >= 3:
                try:
                    # Prepare data for PCA
                    pca_data = self.data[self.numeric_cols].dropna()
                    if len(pca_data) > 10:  # Need at least some samples
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(pca_data)
                        
                        # Perform PCA
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Create PCA plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
                        ax.set_title('PCA: First Two Principal Components')
                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                        
                        # Add feature vectors
                        for i, col in enumerate(self.numeric_cols):
                            ax.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3, 
                                   head_width=0.05, head_length=0.05, fc='red', ec='red')
                            ax.text(pca.components_[0, i]*3.15, pca.components_[1, i]*3.15, col)
                            
                        figures['pca'] = fig
                        
                        if output_dir:
                            fig.savefig(f"{output_dir}/pca.png")
                except Exception as e:
                    logger.warning(f"Error generating PCA visualization: {str(e)}")
            
            # Time series plot if datetime columns exist
            if self.datetime_cols and self.numeric_cols:
                date_col = self.datetime_cols[0]  # Use first datetime column
                for num_col in self.numeric_cols[:3]:  # Limit to first 3 numeric columns
                    try:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        self.data.sort_values(date_col).plot(x=date_col, y=num_col, ax=ax)
                        ax.set_title(f'{num_col} over time')
                        plt.tight_layout()
                        figures[f'timeseries_{num_col}'] = fig
                        
                        if output_dir:
                            fig.savefig(f"{output_dir}/timeseries_{num_col}.png")
                    except Exception as e:
                        logger.warning(f"Error generating time series for {num_col}: {str(e)}")
            
            # Categorical distributions
            if self.categorical_cols:
                for cat_col in self.categorical_cols[:5]:  # Limit to first 5 categorical columns
                    if self.data[cat_col].nunique() < 15:  # Skip high cardinality
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            self.data[cat_col].value_counts().plot(kind='bar', ax=ax)
                            ax.set_title(f'Distribution of {cat_col}')
                            plt.tight_layout()
                            figures[f'categorical_{cat_col}'] = fig
                            
                            if output_dir:
                                fig.savefig(f"{output_dir}/categorical_{cat_col}.png")
                        except Exception as e:
                            logger.warning(f"Error generating categorical plot for {cat_col}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        return figures
    
    def summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of the dataset.
        
        Returns:
            Dictionary containing all analysis results
        """
        report = {
            'dataset_info': {
                'shape': self.data.shape,
                'memory_usage': self.data.memory_usage(deep=True).sum() / (1024*1024),  # MB
                'numeric_columns': self.numeric_cols,
                'categorical_columns': self.categorical_cols,
                'datetime_columns': self.datetime_cols,
                'missing_values': self.data.isnull().sum().to_dict()
            },
            'patterns': self.identify_patterns(),
            'anomalies': self.find_anomalies()
        }
        
        if len(self.numeric_cols) >= 2:
            report['feature_importance'] = self.feature_importance()
            report['clusters'] = self.cluster_analysis()
        
        return report
    
    def print_summary(self, detailed: bool = False):
        """
        Print a human-readable summary of the dataset analysis.
        
        Args:
            detailed: Whether to include detailed statistics
        """
        patterns = self.identify_patterns()
        anomalies = self.find_anomalies()
        
        print("\n=== PATTERN ANALYSIS SUMMARY ===")
        print(f"\nDataset dimensions: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        print(f"Numeric columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols)}")
        print(f"Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols[:5])}" + 
              (f" and {len(self.categorical_cols)-5} more..." if len(self.categorical_cols) > 5 else ""))
        
        # Missing values
        print("\n--- Missing Values ---")
        missing = patterns['missing_values']
        print(f"Total rows with missing values: {missing['total_missing_rows']} ({missing['total_missing_rows']/self.data.shape[0]:.1%})")
        missing_cols = {k: v for k, v in missing['counts'].items() if v > 0}
        if missing_cols:
            print("Columns with missing values:")
            for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {col}: {count} missing ({missing['percentages'][col]:.1f}%)")
        else:
            print("No missing values found.")
        
        # Basic stats
        if not detailed:
            print("\n--- Basic Statistics ---")
            for col in self.numeric_cols[:3]:  # Show only first 3 columns
                stats = patterns['basic_stats'][col]
                print(f"  {col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
            if len(self.numeric_cols) > 3:
                print(f"  ... and {len(self.numeric_cols)-3} more columns")
        
        # Group statistics
        if 'group_statistics' in patterns and patterns['group_statistics']:
            print("\n--- Group Statistics ---")
            for cat_col, stats in list(patterns['group_statistics'].items())[:2]:  # First 2 categorical columns
                print(f"\nGrouped by {cat_col} ({len(stats['counts'])} groups):")
                # Show means for first numeric column by top 3 categories
                if self.numeric_cols:
                    num_col = self.numeric_cols[0]
                    top_groups = sorted(stats['counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"  Means of {num_col} for top groups:")
                    for group, count in top_groups:
                        try:
                            mean_val = stats['means'][num_col][group]
                            print(f"    - {group} ({count} items): {mean_val:.2f}")
                        except (KeyError, TypeError):
                            pass
        
        # Correlations
        if 'strong_correlations' in patterns and patterns['strong_correlations']:
            print("\n--- Strong Correlations ---")
            for col1, col2, corr in patterns['strong_correlations'][:5]:
                print(f"  {col1} and {col2}: {corr:.3f}")
        
        # Anomalies
        if anomalies:
            print("\n--- Anomalies ---")
            for method, results in anomalies.items():
                print(f"\n  Method: {method}")
                for col, result in results.items():
                    print(f"    - {col}: {result.outlier_count} outliers")
                    if result.outlier_count > 0 and result.outlier_count <= 5:
                        print(f"      Values: {list(result.outlier_values.values())}")
        
        # Feature importance
        if len(self.numeric_cols) >= 2:
            print("\n--- Feature Importance ---")
            importance = self.feature_importance()
            for col, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {col}: {score:.4f}")
        
        if detailed:
            # Print more detailed statistical information
            print("\n=== DETAILED STATISTICS ===")
            for col in self.numeric_cols:
                print(f"\nColumn: {col}")
                stats = patterns['basic_stats'][col]
                for stat, value in stats.items():
                    print(f"  {stat}: {value}")
                
                if col in patterns.get('distribution_shape', {}):
                    shape = patterns['distribution_shape'][col]
                    print(f"  Skewness: {shape['skewness']:.3f}")
                    print(f"  Kurtosis: {shape['kurtosis']:.3f}")
            
            # Categorical distributions
            if self.categorical_cols:
                print("\n=== CATEGORICAL DISTRIBUTIONS ===")
                for col in self.categorical_cols[:5]:
                    if col in patterns.get('categorical_cardinality', {}):
                        card = patterns['categorical_cardinality'][col]
                        print(f"\nColumn: {col} (unique: {card['unique_count']})")
                        print("  Top values:")
                        for val, count in card['top_values'].items():
                            print(f"    - {val}: {count}")
    
    def print_patterns(self):
        """Prints identified patterns and anomalies in a readable format."""
        patterns = self.identify_patterns()
        anomalies = self.find_anomalies()
        print("\n=== PATTERN ANALYSIS ===")
        # Basic stats
        if 'basic_stats' in patterns:
            print("\nBasic Stats:")
            for col, stats in patterns['basic_stats'].items():
                print(f"- {col}: {stats}")
        # Distribution shape
        if 'distribution_shape' in patterns:
            print("\nDistribution Shape:")
            for col, shape in patterns['distribution_shape'].items():
                print(f"- {col}: skewness={shape['skewness']:.2f}, kurtosis={shape['kurtosis']:.2f}")
        # Group statistics
        if 'group_statistics' in patterns:
            print("\nGroup Statistics:")
            for cat_col, stats in patterns['group_statistics'].items():
                print(f"- {cat_col}: means={stats['means']}, medians={stats['medians']}")
        # Correlation matrix
        if 'correlation_matrix' in patterns:
            print("\nCorrelation Matrix:")
            print(pd.DataFrame(patterns['correlation_matrix']))
        # Strong correlations
        if 'strong_correlations' in patterns:
            print("\nStrong Correlations:")
            for col1, col2, corr in patterns['strong_correlations']:
                print(f"- {col1} & {col2}: {corr}")
        # Missing values
        if 'missing_values' in patterns:
            print("\nMissing Values:")
            print(patterns['missing_values'])
        # Categorical cardinality
        if 'categorical_cardinality' in patterns:
            print("\nCategorical Cardinality:")
            print(patterns['categorical_cardinality'])
        # Anomalies
        print("\nAnomalies:")
        for method, result in anomalies.items():
            print(f"- {method}:")
            if method == 'isolation_forest':
                iso = result.get('multidimensional', None)
                if iso is None:
                    print("  Not enough data for isolation forest or method skipped.")
                else:
                    print(f"  multidimensional: {iso.outlier_count} outliers, bounds={iso.bounds}")
            else:
                for col, res in result.items():
                    print(f"  {col}: {res.outlier_count} outliers, bounds={res.bounds}")
    
    def _find_iqr_anomalies(self, factor: float = 1.5) -> Dict[str, AnomalyResult]:
        """
        Find outliers using the IQR method.
        
        Args:
            factor: IQR multiplier for determining outlier boundaries
            
        Returns:
            Dictionary of outliers by column
        """
        results = {}
        for col in self.numeric_cols:
            try:
                series = self.data[col].dropna()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                
                if abs(iqr) < 1e-10:  # Avoid division by zero or small values
                    continue
                    
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                
                outlier_series = self.data[(self.data[col] < lower) | (self.data[col] > upper)]
                outlier_indices = outlier_series.index.tolist()
                outlier_values = outlier_series[col].to_dict()
                
                results[col] = AnomalyResult(
                    outlier_count=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    outlier_values=outlier_values,
                    bounds=(float(lower), float(upper)),
                    method='iqr'
                )
            except Exception as e:
                logger.warning(f"Error finding IQR anomalies for {col}: {str(e)}")
        
        return results
    
    def _find_zscore_anomalies(self, threshold: float = 3.0) -> Dict[str, AnomalyResult]:
        """
        Find outliers using Z-score method.
        
        Args:
            threshold: Z-score threshold for determining outliers
            
        Returns:
            Dictionary of outliers by column
        """
        results = {}
        for col in self.numeric_cols:
            try:
                series = self.data[col].dropna()
                if len(series) < 2:  # Need at least 2 values for std
                    continue
                    
                mean = series.mean()
                std = series.std()
                
                if abs(std) < 1e-10:  # Avoid division by zero or small values
                    continue
                
                z_scores = (series - mean) / std
                outlier_indices = self.data[abs(z_scores) > threshold].index.tolist()
                outlier_values = self.data.loc[outlier_indices, col].to_dict()
                
                results[col] = AnomalyResult(
                    outlier_count=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    outlier_values=outlier_values,
                    bounds=(float(mean - threshold * std), float(mean + threshold * std)),
                    method='zscore'
                )
            except Exception as e:
                logger.warning(f"Error finding Z-score anomalies for {col}: {str(e)}")
        
        return results
    
    def _find_modified_zscore_anomalies(self, threshold: float = 3.5) -> Dict[str, AnomalyResult]:
        """
        Find outliers using modified Z-score (more robust to outliers).
        
        Args:
            threshold: Modified Z-score threshold for determining outliers
            
        Returns:
            Dictionary of outliers by column
        """
        results = {}
        for col in self.numeric_cols:
            try:
                series = self.data[col].dropna()
                if len(series) < 3:  # Need at least 3 values for MAD
                    continue
                
                median = series.median()
                # Median Absolute Deviation
                mad = np.median(np.abs(series - median))
                
                if abs(mad) < 1e-10:  # Avoid division by zero or small values
                    continue
                
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_indices = self.data[abs(modified_z_scores) > threshold].index.tolist()
                outlier_values = self.data.loc[outlier_indices, col].to_dict()
                
                lower_bound = median - (threshold * mad / 0.6745)
                upper_bound = median + (threshold * mad / 0.6745)
                
                results[col] = AnomalyResult(
                    outlier_count=len(outlier_indices),
                    outlier_indices=outlier_indices,
                    outlier_values=outlier_values,
                    bounds=(float(lower_bound), float(upper_bound)),
                    method='modified_zscore'
                )
            except Exception as e:
                logger.warning(f"Error finding modified Z-score anomalies for {col}: {str(e)}")
        
        return results
    
    def _find_isolation_forest_anomalies(self, contamination: float = 0.05) -> Dict[str, AnomalyResult]:
        """
        Find outliers using Isolation Forest algorithm (multidimensional).
        
        Args:
            contamination: Expected proportion of outliers in the data
            
        Returns:
            Dictionary with multidimensional outliers
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            # Prepare data - use only numeric columns with no missing values
            complete_rows = self.data[self.numeric_cols].dropna()
            if len(complete_rows) < 10:  # Need sufficient data
                logger.warning("Not enough complete rows for Isolation Forest")
                return {}
                
            # Normalize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(complete_rows)
            
            # Train isolation forest
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Predict outliers (-1 for outliers, 1 for inliers)
            predictions = model.fit_predict(scaled_data)
            outlier_indices = complete_rows.index[predictions == -1].tolist()
            
            # Calculate anomaly scores (higher is more anomalous)
            scores = model.decision_function(scaled_data)
            outlier_scores = {idx: -float(scores[i]) for i, idx in enumerate(complete_rows.index) if predictions[i] == -1}
            
            # Store result as multidimensional anomalies
            result = AnomalyResult(
                outlier_count=len(outlier_indices),
                outlier_indices=outlier_indices,
                outlier_values=outlier_scores,  # Scores instead of values
                bounds=(float(np.min(scores)), float(np.max(scores))),
                method='isolation_forest'
            )
            
            return {'multidimensional': result}
            
        except Exception as e:
            logger.warning(f"Error in isolation forest anomaly detection: {str(e)}")
            return {}
    
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """
        Determine the optimal number of clusters using the elbow method.
        
        Args:
            data: Standardized data array
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.cluster import KMeans
        
        max_clusters = min(max_clusters, data.shape[0] // 5, 10)  # Sensible upper limit
        
        # Calculate sum of squared distances for different k values
        inertia = []
        k_values = range(1, max_clusters + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)
        
        # Find elbow point using the kneedle algorithm if available
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator(
                k_values, inertia, curve='convex', direction='decreasing'
            )
            optimal_k = kneedle.elbow
            
            if optimal_k is None:
                # Fallback if no clear elbow
                optimal_k = 2
        except ImportError:
            # Manual elbow detection
            if len(inertia) <= 2:
                return 2
                
            # Calculate the second derivative
            deltas = np.diff(inertia, 2)
            optimal_idx = np.argmax(deltas) + 2  # +2 because of double diff
            optimal_k = k_values[optimal_idx]
        
        # Ensure we return a valid k value
        if optimal_k is None or optimal_k < 2:
            optimal_k = 2
        elif optimal_k > max_clusters:
            optimal_k = max_clusters
            
        return optimal_k
    
    def _calculate_cluster_feature_importance(self, kmeans, feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance for clustering.
        
        Args:
            kmeans: Fitted KMeans model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        # Calculate how much each feature contributes to cluster separation
        centroids = kmeans.cluster_centers_
        overall_range = np.max(centroids, axis=0) - np.min(centroids, axis=0)
        
        # Normalize the range contributions
        total_range = np.sum(overall_range)
        if total_range > 0:
            importance = overall_range / total_range
        else:
            importance = np.ones_like(overall_range) / len(overall_range)
        
        return {name: float(imp) for name, imp in zip(feature_names, importance)}
    
    def _analyze_time_patterns(self) -> Dict[str, Any]:
        """
        Analyze time-based patterns in the dataset.
        
        Returns:
            Dictionary of time-based patterns
        """
        if not self.datetime_cols:
            return {}
            
        time_patterns = {}
        date_col = self.datetime_cols[0]  # Use first datetime column
        
        try:
            # Ensure datetime format
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            
            # Extract datetime components
            time_df = pd.DataFrame()
            time_df['hour'] = self.data[date_col].dt.hour
            time_df['day'] = self.data[date_col].dt.day
            time_df['weekday'] = self.data[date_col].dt.weekday
            time_df['month'] = self.data[date_col].dt.month
            time_df['year'] = self.data[date_col].dt.year
            
            # Count occurrences by unit
            time_patterns['counts'] = {
                'hourly': time_df['hour'].value_counts().to_dict(),
                'daily': time_df['day'].value_counts().to_dict(),
                'weekday': time_df['weekday'].value_counts().to_dict(),
                'monthly': time_df['month'].value_counts().to_dict(),
                'yearly': time_df['year'].value_counts().to_dict()
            }
            
            # Time-based statistics for numeric columns
            time_patterns['time_statistics'] = {}
            for num_col in self.numeric_cols:
                # Monthly averages
                monthly_avg = self.data.groupby(self.data[date_col].dt.month)[num_col].mean().to_dict()
                time_patterns['time_statistics'][num_col] = {'monthly_avg': monthly_avg}
                
                # Detect seasonality and trends
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    # Prepare time series data
                    ts_data = self.data.set_index(date_col)[num_col].dropna().sort_index()
                    if len(ts_data) >= 12:  # Need sufficient data points
                        # Try to find appropriate seasonality period
                        freq = pd.infer_freq(ts_data.index)
                        if freq:
                            # Decompose time series
                            result = seasonal_decompose(ts_data, model='additive')
                            time_patterns['time_statistics'][num_col]['trend'] = result.trend.dropna().to_dict()
                            time_patterns['time_statistics'][num_col]['seasonal'] = result.seasonal.dropna().to_dict()
                except ImportError:
                    logger.info("statsmodels not available for time series decomposition")
                except Exception as e:
                    logger.warning(f"Error in time series decomposition for {num_col}: {str(e)}")
            
            # Calculate intervals and detect regularity
            if len(self.data[date_col]) > 1:
                sorted_dates = sorted(self.data[date_col].dropna())
                intervals = [(sorted_dates[i+1] - sorted_dates[i]).total_seconds() 
                             for i in range(len(sorted_dates)-1)]
                
                if intervals:
                    time_patterns['intervals'] = {
                        'mean_interval_seconds': float(np.mean(intervals)),
                        'std_interval_seconds': float(np.std(intervals)),
                        'regularity': 1 - min(1, float(np.std(intervals) / (np.mean(intervals) + 1e-10)))
                    }
            
        except Exception as e:
            logger.warning(f"Error analyzing time patterns: {str(e)}")
            
        return time_patterns
        
    def detect_data_quality_issues(self) -> Dict[str, Any]:
        """
        Detect various data quality issues in the dataset.
        
        Returns:
            Dictionary of data quality issues
        """
        issues = {}
        
        # Check for missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        issues['missing_values'] = {col: {'count': int(count), 'percentage': float(missing_pct[col])} 
                                   for col, count in missing.items() if count > 0}
        
        # Check for duplicate rows
        duplicates = self.data.duplicated()
        duplicate_count = duplicates.sum()
        issues['duplicate_rows'] = {
            'count': int(duplicate_count),
            'percentage': float(round(duplicate_count / len(self.data) * 100, 2)),
            'indices': self.data[duplicates].index.tolist()
        }
        
        # Check for constant columns
        constant_cols = []
        for col in self.data.columns:
            if self.data[col].nunique() <= 1:
                constant_cols.append(col)
        issues['constant_columns'] = constant_cols
        
        # Check for high cardinality categorical columns
        high_card_cols = []
        for col in self.categorical_cols:
            ratio = self.data[col].nunique() / len(self.data)
            if ratio > 0.9:  # Over 90% unique values
                high_card_cols.append({
                    'column': col,
                    'unique_count': self.data[col].nunique(),
                    'uniqueness_ratio': float(round(ratio, 3))
                })
        issues['high_cardinality_columns'] = high_card_cols
        
        # Check for highly skewed numeric columns
        skewed_cols = []
        for col in self.numeric_cols:
            try:
                skewness = float(stats.skew(self.data[col].dropna()))
                if abs(skewness) > 3:  # Highly skewed
                    skewed_cols.append({
                        'column': col,
                        'skewness': round(skewness, 3)
                    })
            except:
                pass
        issues['highly_skewed_columns'] = skewed_cols
        
        # Check for near-zero variance columns
        near_zero_var = []
        for col in self.numeric_cols:
            variance = self.data[col].var()
            if abs(variance) < 1e-6:
                near_zero_var.append(col)
        issues['near_zero_variance_columns'] = near_zero_var
        
        # Check for potential mixed data types
        mixed_types = []
        for col in self.categorical_cols:
            # Check if column might contain numeric data
            try:
                numeric_count = pd.to_numeric(self.data[col], errors='coerce').notna().sum()
                if numeric_count > 0 and numeric_count < len(self.data):
                    mixed_types.append({
                        'column': col,
                        'numeric_count': int(numeric_count),
                        'percentage': float(round(numeric_count / len(self.data) * 100, 2))
                    })
            except:
                pass
        issues['potential_mixed_types'] = mixed_types
        
        return issues
    
    def detect_complex_patterns(self) -> Dict[str, Any]:
        """
        Perform more advanced pattern detection in the dataset.
        
        Returns:
            Dictionary of detected complex patterns
        """
        patterns = {}
        
        # Bimodal/multimodal distribution detection
        multimodal = []
        for col in self.numeric_cols:
            try:
                data = self.data[col].dropna()
                if len(data) < 100:  # Need sufficient data
                    continue
                
                # Use Kernel Density Estimation to find modes
                from scipy.signal import argrelextrema
                from scipy import stats
                
                kde = stats.gaussian_kde(data)
                x = np.linspace(min(data), max(data), 1000)
                y = kde(x)
                
                # Find local maxima
                local_max_indices = argrelextrema(y, np.greater)[0]
                
                if len(local_max_indices) > 1:
                    modes = [float(x[idx]) for idx in local_max_indices]
                    multimodal.append({
                        'column': col,
                        'mode_count': len(modes),
                        'modes': modes
                    })
            except Exception as e:
                logger.debug(f"Error detecting multimodality for {col}: {str(e)}")
        
        patterns['multimodal_distributions'] = multimodal
        
        # Look for cyclic patterns in time series data
        if self.datetime_cols:
            try:
                from scipy import fftpack
                
                cyclic_patterns = []
                date_col = self.datetime_cols[0]
                
                for num_col in self.numeric_cols:
                    try:
                        # Prepare time series (needs to be uniform)
                        ts = self.data.sort_values(date_col).set_index(date_col)[num_col].dropna()
                        if len(ts) < 24:  # Need sufficient data
                            continue
                            
                        # If timestamps are irregular, resample to regular intervals
                        if pd.infer_freq(ts.index) is None:
                            # Determine appropriate frequency
                            ts_range = (ts.index.max() - ts.index.min()).total_seconds()
                            n_points = len(ts)
                            avg_interval = ts_range / n_points
                            
                            # Choose frequency based on average interval
                            if avg_interval < 60:  # Less than a minute
                                freq = '1S'  # 1 second
                            elif avg_interval < 3600:  # Less than an hour
                                freq = '1min'
                            elif avg_interval < 86400:  # Less than a day
                                freq = '1H'
                            else:
                                freq = '1D'
                                
                            # Resample
                            ts = ts.resample(freq).mean().interpolate(method='linear')
                        
                        # Apply FFT to find cyclical patterns
                        y = ts.values
                        n = len(y)
                        dt = 1  # normalized time step
                        
                        # Compute FFT
                        yf = fftpack.fft(y - np.mean(y))
                        freqs = fftpack.fftfreq(n, dt)
                        
                        # Find dominant frequencies (exclude zero frequency)
                        power = abs(yf)**2
                        threshold = np.percentile(power[1:n//2], 95)  # Top 5% power
                        
                        dominant_idx = np.where(power[1:n//2] > threshold)[0] + 1
                        dominant_freqs = [float(freqs[i]) for i in dominant_idx]
                        dominant_periods = [float(1/f) if f != 0 else float('inf') for f in dominant_freqs]
                        
                        if dominant_periods:
                            cyclic_patterns.append({
                                'column': num_col,
                                'periods': dominant_periods[:3],  # Top 3 periods
                                'power': [float(power[i]) for i in dominant_idx][:3]
                            })
                    except Exception as e:
                        logger.debug(f"Error finding cyclical patterns for {num_col}: {str(e)}")
                        
                patterns['cyclic_patterns'] = cyclic_patterns
            except ImportError:
                logger.info("scipy.fftpack not available for frequency analysis")
        
        # Look for clusters/segments that are high-dimensional features of interest
        # (Already implemented in cluster_analysis method)
        
        return patterns
