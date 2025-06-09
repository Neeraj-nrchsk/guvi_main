import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        self.original_data = data.copy()
        self.data = data.copy()
        self._species_encoder = None
        self._scaler = None  # Store scaler object for consistency

    def select_features(self, 
                      keep: Optional[List[str]] = None,
                      exclude: List[str] = ['Id'],
                      inplace: bool = False) -> pd.DataFrame:
        """
        Select features for modeling
        
        Args:
            keep: List of specific columns to keep (None keeps all)
            exclude: Columns to exclude (default: ['Id'])
            inplace: Modify the original data or return a copy
            
        Returns:
            Modified DataFrame
        """
        df = self.data if inplace else self.data.copy()
        
        # Validate columns exist
        all_cols = set(df.columns)
        invalid = set(exclude) - all_cols
        if invalid:
            logger.warning(f"Columns not found for exclusion: {invalid}")
            
        # Apply selection
        if keep:
            invalid_keep = set(keep) - all_cols
            if invalid_keep:
                raise ValueError(f"Columns to keep not found: {invalid_keep}")
            df = df[keep]
        else:
            df = df.drop(columns=[c for c in exclude if c in df.columns])
            
        logger.info(f"Selected features: {list(df.columns)}")
        return df

    def engineer_features(self,
                        create_ratios: bool = True,
                        create_areas: bool = True,
                        encode_species: bool = True,
                        inplace: bool = False) -> pd.DataFrame:
        """
        Create meaningful features for iris classification
        
        Args:
            create_ratios: Add length/width ratios
            create_areas: Add sepal/petal areas
            encode_species: Encode target labels
            inplace: Modify original data
            
        Returns:
            DataFrame with engineered features
        """
        df = self.data if inplace else self.data.copy()
        
        # 1. Create biological ratio features
        if create_ratios:
            df['sepal_ratio'] = df['SepalLengthCm'] / (df['SepalWidthCm'] + 1e-9)
            df['petal_ratio'] = df['PetalLengthCm'] / (df['PetalWidthCm'] + 1e-9)
            logger.info("Added ratio features")
            
        # 2. Create size estimation features
        if create_areas:
            df['sepal_area'] = df['SepalLengthCm'] * df['SepalWidthCm']
            df['petal_area'] = df['PetalLengthCm'] * df['PetalWidthCm']
            logger.info("Added area features")
            
        # 3. Encode target variable
        if encode_species and 'Species' in df.columns:
            if self._species_encoder is None:
                self._species_encoder = LabelEncoder()
                df['Species_encoded'] = self._species_encoder.fit_transform(df['Species'])
            else:
                df['Species_encoded'] = self._species_encoder.transform(df['Species'])
            logger.info("Encoded species labels")
            
        return df

    def scale_features(self, 
                     method: str = 'standard',
                     exclude: List[str] = ['Species', 'Species_encoded'],
                     inplace: bool = False) -> pd.DataFrame:
        """
        Scale numeric features using specified method
        
        Args:
            method: 'standard' (Z-score) or 'minmax' (0-1 scaling)
            exclude: Columns to exclude from scaling
            inplace: Modify original data
            
        Returns:
            Scaled DataFrame
        """
        df = self.data if inplace else self.data.copy()
        
        # Identify numeric columns to scale
        numeric_cols = df.select_dtypes(include=np.number).columns
        scale_cols = [col for col in numeric_cols if col not in exclude]
        
        if not scale_cols:
            logger.warning("No numeric columns to scale")
            return df
            
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Choose 'standard' or 'minmax'")
        
        # Fit and transform
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        self._scaler = scaler  # Store for potential inverse transforms
        logger.info(f"Applied {method} scaling to: {scale_cols}")
        
        return df

    def get_feature_descriptions(self) -> dict:
        """Get descriptions of engineered features"""
        return {
            'sepal_ratio': 'Sepal length-to-width ratio',
            'petal_ratio': 'Petal length-to-width ratio',
            'sepal_area': 'Sepal area (length * width) in cm²',
            'petal_area': 'Petal area (length * width) in cm²',
            'Species_encoded': 'Encoded species labels (0: setosa, 1: versicolor, 2: virginica)'
        }

    def restore_original(self) -> None:
        """Restore data to original state"""
        self.data = self.original_data.copy()
        logger.info("Restored original data")

# Usage Example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Iris.csv")
    
    # Initialize feature engineer
    fe = FeatureEngineering(df)
    
    # Feature selection
    df_selected = fe.select_features(exclude=['Id'])
    
    # Feature engineering
    df_engineered = fe.engineer_features()
    
    # Feature scaling
    df_scaled = fe.scale_features(method='standard')
    
    print("\nEngineered Features:")
    print(fe.get_feature_descriptions())
    
    print("\nScaled Features Summary:")
    print(df_scaled[['SepalLengthCm', 'SepalWidthCm', 
                   'PetalLengthCm', 'PetalWidthCm',
                   'sepal_ratio', 'petal_ratio']].describe().round(2))
    
    print("\nSample Data:")
    print(df_scaled[['sepal_ratio', 'petal_area', 'Species_encoded']].head())