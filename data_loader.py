import pandas as pd
import logging
from typing import Optional, Union, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize DataLoader with optional file path.
        
        Args:
            filepath: Optional path to dataset file
        """
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None
        self._expected_columns = {
            'SepalLengthCm', 'SepalWidthCm', 
            'PetalLengthCm', 'PetalWidthCm', 'Species'
        }

    def load_data(self, 
                 source: str = 'local',
                 file_type: str = 'csv',
                 **kwargs) -> pd.DataFrame:
        """
        Load Iris dataset from various sources.
        
        Args:
            source: Data source ('local', 'kaggle', 'url')
            file_type: File format ('csv', 'excel', 'json', 'parquet')
            kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: For unsupported sources or file types
            FileNotFoundError: For local file not found
        """
        loaders = {
            'local': self._load_local,
            'kaggle': self._load_kaggle,
            'url': self._load_from_url
        }
        
        if source not in loaders:
            raise ValueError(f"Unsupported source: {source}. Choose from {list(loaders.keys())}")
            
        try:
            self.data = loaders[source](file_type, **kwargs)
            self._validate_dataset()
            logger.info("Dataset loaded successfully with %d rows", len(self.data))
            return self.data
        except Exception as e:
            logger.error("Failed to load dataset: %s", str(e))
            raise

    def _load_local(self, 
                   file_type: str,
                   **kwargs) -> pd.DataFrame:
        """Load local dataset file"""
        if not self.filepath:
            self.filepath = 'Iris.csv'  # Default fallback
            
        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        reader = {
            'csv': pd.read_csv,
            'excel': pd.read_excel,
            'json': pd.read_json,
            'parquet': pd.read_parquet
        }.get(file_type.lower())

        if not reader:
            raise ValueError(f"Unsupported file type: {file_type}")

        return reader(path, **kwargs)

    def _load_kaggle(self,
                    file_type: str = 'csv',
                    dataset: str = 'uciml/iris',
                    file_name: str = 'Iris.csv',
                    **kwargs) -> pd.DataFrame:
        """Load dataset from Kaggle using kagglehub"""
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
        except ImportError:
            raise ImportError("kagglehub package required. Install with 'pip install kagglehub'")

        try:
            logger.info("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download(dataset)
            logger.info("Dataset downloaded to: %s", path)
            
            return kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                dataset,
                file_name,
                file_type=file_type,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Kaggle download failed: {str(e)}")

    def _load_from_url(self,
                      file_type: str,
                      url: str,
                      **kwargs) -> pd.DataFrame:
        """Load dataset from URL"""
        try:
            reader = {
                'csv': pd.read_csv,
                'excel': pd.read_excel,
                'json': pd.read_json,
                'parquet': pd.read_parquet
            }[file_type.lower()]
        except KeyError:
            raise ValueError(f"Unsupported file type for URL: {file_type}")

        return reader(url, **kwargs)

    def _validate_dataset(self) -> None:
        """Validate dataset structure and content"""
        if self.data is None:
            raise ValueError("No data loaded for validation")
            
        missing_cols = self._expected_columns - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
            
        if self.data.empty:
            raise ValueError("Loaded dataset is empty")
            
        logger.info("Dataset validation passed")

    def reload_data(self, **kwargs) -> pd.DataFrame:
        """Reload data using current configuration"""
        return self.load_data(**kwargs)

    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """Get sample data from loaded dataset"""
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.sample(n)

    def save_data(self, 
                 output_path: str,
                 file_type: str = 'csv',
                 **kwargs) -> None:
        """
        Save loaded data to specified format
        
        Args:
            output_path: Path to save file
            file_type: Output format ('csv', 'excel', etc.)
        """
        if self.data is None:
            raise ValueError("No data to save")

        saver = {
            'csv': self.data.to_csv,
            'excel': self.data.to_excel,
            'json': self.data.to_json,
            'parquet': self.data.to_parquet
        }.get(file_type.lower())

        if not saver:
            raise ValueError(f"Unsupported output format: {file_type}")

        saver(output_path, **kwargs)
        logger.info("Data saved to %s", output_path)

    @property
    def data_summary(self) -> Dict:
        """Get basic data summary"""
        if self.data is None:
            return {}
            
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isna().sum().to_dict(),
            'dtypes': self.data.dtypes.astype(str).to_dict()
        }