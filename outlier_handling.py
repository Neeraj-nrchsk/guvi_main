import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutlierHandler:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.original_data = data.copy()
        self.data = data.copy()
        self.report: Dict = {}
        self._min_std = 1e-8  # Minimum standard deviation threshold

    def detect(self,
              method: str = 'zscore',
              columns: Optional[List[str]] = None,
              group_by: str = 'Species',
              **kwargs) -> Dict:
        """
        Detect outliers using specified method with stability improvements
        """
        if columns is None:
            columns = self._numeric_columns()
            
        if group_by and group_by in self.data.columns:
            return self._grouped_outlier_detection(method, columns, group_by, **kwargs)
            
        return self._detect(method, columns, **kwargs)

    def _numeric_columns(self) -> List[str]:
        """Get list of numeric columns excluding potential ID columns"""
        return [col for col in self.data.select_dtypes(include=np.number).columns 
                if not col.lower().startswith('id')]

    def _detect(self,
               method: str,
               columns: List[str],
               **kwargs) -> Dict:
        """Main detection logic with variance checks"""
        detectors = {
            'zscore': self._zscore_detection,
            'iqr': self._iqr_detection,
            'isolation_forest': self._isolation_forest_detection
        }
        return detectors[method](columns, **kwargs)

    def _zscore_detection(self,
                         columns: List[str],
                         threshold: float = 3.0) -> Dict:
        """Z-score detection with variance stability checks"""
        outliers = {}
        for col in columns:
            if self.data[col].std() < self._min_std:
                logger.warning(f"Skipping z-score for {col} (low variance)")
                outliers[col] = {'indices': [], 'count': 0, 'threshold': threshold}
                continue
                
            try:
                z_scores = stats.zscore(self.data[col])
                mask = np.abs(z_scores) > threshold
                outliers[col] = {
                    'indices': self.data[mask].index.tolist(),
                    'count': mask.sum(),
                    'threshold': threshold
                }
            except Exception as e:
                logger.error(f"Z-score failed for {col}: {str(e)}")
                outliers[col] = {'indices': [], 'count': 0, 'error': str(e)}
                
        self.report['zscore'] = outliers
        return outliers

    def _iqr_detection(self,
                      columns: List[str],
                      factor: float = 1.5) -> Dict:
        """IQR detection with enhanced stability"""
        outliers = {}
        for col in columns:
            try:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                
                if iqr < self._min_std:
                    logger.warning(f"Skipping IQR for {col} (low variability)")
                    outliers[col] = {'indices': [], 'count': 0, 'bounds': (None, None)}
                    continue
                    
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                mask = (self.data[col] < lower) | (self.data[col] > upper)
                outliers[col] = {
                    'indices': self.data[mask].index.tolist(),
                    'count': mask.sum(),
                    'bounds': (lower, upper)
                }
            except Exception as e:
                logger.error(f"IQR failed for {col}: {str(e)}")
                outliers[col] = {'indices': [], 'count': 0, 'error': str(e)}
                
        self.report['iqr'] = outliers
        return outliers

    def _isolation_forest_detection(self,
                                   columns: List[str],
                                   contamination: float = 0.05) -> Dict:
        """Multivariate outlier detection using Isolation Forest"""
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(self.data[columns])
        mask = preds == -1
        return {
            'indices': self.data[mask].index.tolist(),
            'count': mask.sum(),
            'contamination': contamination
        }

    def handle(self,
              strategy: str = 'remove',
              method: str = 'zscore',
              columns: Optional[List[str]] = None,
              **kwargs) -> pd.DataFrame:
        """
        Handle outliers with improved validation
        """
        try:
            if strategy == 'remove':
                return self._remove_outliers(method, columns, **kwargs)
            elif strategy == 'cap':
                return self._cap_outliers(method, columns, **kwargs)
            elif strategy == 'impute':
                return self._impute_outliers(method, columns, **kwargs)
            return self.data
        except Exception as e:
            logger.error(f"Outlier handling failed: {str(e)}")
            self.restore_original()
            raise

    def _remove_outliers(self,
                        method: str,
                        columns: Optional[List[str]],
                        **kwargs) -> pd.DataFrame:
        """Remove outliers with pre-check validation"""
        if self.data.empty:
            logger.warning("Empty dataframe - nothing to process")
            return self.data
            
        outliers = self.detect(method, columns, **kwargs)
        mask = np.ones(len(self.data), dtype=bool)
        
        for col, info in outliers.items():
            if 'indices' in info:
                mask[info['indices']] = False
            elif 'error' in info:
                logger.warning(f"Skipping {col} due to previous error")
                
        cleaned = self.data[mask].reset_index(drop=True)
        logger.info(f"Removed {len(self.data) - len(cleaned)} outliers")
        self.data = cleaned
        return self.data

    def _cap_outliers(self,
                     method: str,
                     columns: Optional[List[str]],
                     **kwargs) -> pd.DataFrame:
        """Cap outliers to detection bounds or z-score limits"""
        if columns is None:
            columns = self._numeric_columns()
        outliers = self.detect(method, columns, **kwargs)
        df = self.data.copy()
        for col, info in outliers.items():
            if method == 'iqr' and 'bounds' in info and info['bounds'][0] is not None:
                lower, upper = info['bounds']
                df[col] = df[col].clip(lower, upper)
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
        self.data = df
        return self.data

    def visualize(self,
                 column: str,
                 method: str = 'boxplot') -> None:
        """Enhanced visualization with biological context"""
        if column not in self.data.columns:
            logger.error(f"Column {column} not found")
            return
            
        plt.figure(figsize=(12, 6))
        
        if method == 'boxplot':
            self.data.boxplot(column=column)
            plt.title(f"Distribution of {column}")
            plt.xticks(rotation=45)
            
        elif method == 'scatter' and 'Species' in self.data.columns:
            for species, group in self.data.groupby('Species'):
                plt.scatter(group[column], group.index, label=species, alpha=0.7)
            plt.title(f"{column} Distribution by Species")
            plt.ylabel("Data Index")
            plt.legend()
            
        plt.tight_layout()
        plt.show()

    def get_report(self) -> Dict:
        """Get report with additional summary metrics"""
        report = self.report.copy()
        report['summary'] = {
            'original_shape': self.original_data.shape,
            'current_shape': self.data.shape,
            'rows_removed': len(self.original_data) - len(self.data)
        }
        return report

    def restore_original(self) -> None:
        """Restore data with validation"""
        self.data = self.original_data.copy()
        logger.info("Data restored to original state")
        self.report = {}

    def _grouped_outlier_detection(self,
                                  method: str,
                                  columns: List[str],
                                  group_by: str,
                                  **kwargs) -> Dict:
        """Detect outliers within each group"""
        results = {}
        for group, df_group in self.data.groupby(group_by):
            handler = OutlierHandler(df_group)
            results[group] = handler._detect(method, columns, **kwargs)
        self.report['grouped_outliers'] = results
        return results