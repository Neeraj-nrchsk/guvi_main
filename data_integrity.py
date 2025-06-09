import pandas as pd
from typing import Dict
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class DataIntegrity:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        self.original_data = data.copy()
        self.data = data.copy()
        self.label_encoder = LabelEncoder()
        self.integrity_report: Dict = {}
        
        if 'Species' in self.data.columns:
            self._encode_species()

    def _encode_species(self) -> None:
        """Encode species labels for model-based operations"""
        self.data['Species_encoded'] = self.label_encoder.fit_transform(self.data['Species'])

    def check_integrity(self, verbose: bool = False) -> Dict:
        """Comprehensive data quality assessment for Iris dataset"""
        report = {
            'missing_values': self._check_missing_values(),
            'duplicates': self._check_duplicates(),
            'data_types': self._check_data_types(),
            'biological_validity': self._check_biological_constraints(),
            'species_quality': self._check_species_quality(),
            'measurement_stats': self._get_measurement_statistics()
        }
        
        self.integrity_report = report
        if verbose:
            self._print_report()
        return report

    def _check_missing_values(self) -> Dict:
        """Analyze missing values in the dataset"""
        missing_counts = self.data.isnull().sum()
        return {
            'counts': missing_counts.to_dict(),
            'percentages': (missing_counts / len(self.data) * 100).round(2).to_dict()
        }

    def _check_duplicates(self) -> Dict:
        """Identify and analyze duplicate rows"""
        dup_count = self.data.duplicated().sum()
        return {
            'count': dup_count,
            'percentage': round((dup_count / len(self.data)) * 100, 2)
        }

    def _check_data_types(self) -> Dict:
        """Verify column data types"""
        return self.data.dtypes.astype(str).to_dict()

    def _check_biological_constraints(self) -> Dict:
        """Validate biological plausibility of measurements"""
        return {
            'valid_sepal_ratio': ((self.data['SepalLengthCm'] / self.data['SepalWidthCm']) < 10).all(),
            'valid_petal_ratio': ((self.data['PetalLengthCm'] / self.data['PetalWidthCm']) < 10).all(),
            'positive_measurements': (self.data[['SepalLengthCm', 'SepalWidthCm', 
                                               'PetalLengthCm', 'PetalWidthCm']] > 0).all().all()
        }

    def _check_species_quality(self) -> Dict:
        """Validate species information"""
        valid_species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        return {
            'valid_species': self.data['Species'].isin(valid_species).all(),
            'distribution': self.data['Species'].value_counts().to_dict()
        }

    def _get_measurement_statistics(self) -> Dict:
        """Generate statistical summary for measurements"""
        cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        return {
            col: {
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'mean': self.data[col].mean().round(2),
                'std': self.data[col].std().round(2),
                'iqr': (self.data[col].quantile(0.75) - self.data[col].quantile(0.25)).round(2)
            } for col in cols
        }

    def clean_data(self, 
                 handle_missing: str = 'knn',
                 remove_duplicates: bool = True,
                 inplace: bool = False) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning
        handle_missing: 'knn' or 'model_based'
        """
        df = self.data if inplace else self.data.copy()
        
        if remove_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)
            
        if handle_missing:
            df = self._handle_missing_values(df, strategy=handle_missing)
            
        df = self._enforce_biological_constraints(df)
        df = self._enforce_measurement_precision(df)
        
        if not inplace:
            return df
        self.data = df
        return self.data

    def _handle_missing_values(self, 
                             df: pd.DataFrame,
                             strategy: str) -> pd.DataFrame:
        """Handle missing values using specified strategy"""
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        
        if strategy == 'knn':
            df[features] = KNNImputer().fit_transform(df[features])
        elif strategy == 'model_based':
            for col in features:
                missing = df[col].isnull()
                if missing.any():
                    model = RandomForestRegressor()
                    X_train = df[~missing][features].drop(col, axis=1)
                    y_train = df[~missing][col]
                    model.fit(X_train, y_train)
                    df.loc[missing, col] = model.predict(df[missing][features].drop(col, axis=1))
        
        if 'Species_encoded' in df.columns and df['Species_encoded'].isnull().any():
            df = self._impute_species(df)
            
        return df

    def _impute_species(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing species using measurement features"""
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        missing = df['Species_encoded'].isnull()
        
        clf = RandomForestClassifier()
        clf.fit(df[~missing][features], df[~missing]['Species_encoded'])
        df.loc[missing, 'Species_encoded'] = clf.predict(df[missing][features])
        df['Species'] = self.label_encoder.inverse_transform(df['Species_encoded'])
        return df.drop(columns='Species_encoded')

    def _enforce_biological_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure measurements follow biological constraints"""
        mask = (
            (df['PetalLengthCm'] <= df['SepalLengthCm'] * 1.5) &
            (df['PetalWidthCm'] <= df['SepalWidthCm'] * 1.5) &
            (df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] > 0).all(axis=1)
        )
        return df[mask].reset_index(drop=True)

    def _enforce_measurement_precision(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure measurement precision of 1 decimal place"""
        cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        df[cols] = df[cols].round(1)
        return df

    def _print_report(self) -> None:
        """Print formatted integrity report"""
        report = self.integrity_report
        print("\n" + " IRIS DATA QUALITY REPORT ".center(60, '='))
        print("\nMissing Values:")
        for col, count in report['missing_values']['counts'].items():
            print(f"- {col}: {count} ({report['missing_values']['percentages'][col]}%)")
            
        print("\nBiological Validity:")
        for k, v in report['biological_validity'].items():
            print(f"- {k.replace('_', ' ').title()}: {v}")
            
        print("\nSpecies Quality:")
        print(f"- All Valid Species: {report['species_quality']['valid_species']}")
        for species, count in report['species_quality']['distribution'].items():
            print(f"  {species}: {count}")

    def restore_original(self) -> None:
        """Restore dataset to original state"""
        self.data = self.original_data.copy()

    def ensure_consistency(self, inplace: bool = False) -> pd.DataFrame:
        """
        Ensures Iris dataset consistency:
        - Valid species names
        - Positive measurements
        - Measurement precision
        - Biological ratios
        - Type consistency
        """
        df = self.data if inplace else self.data.copy()
        # Validate species
        valid_species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        df = df[df['Species'].isin(valid_species)]
        # Validate measurements
        measurement_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        df = df[(df[measurement_cols] > 0).all(axis=1)]
        # Ensure precision consistency
        for col in measurement_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(1)
        # Biological constraints
        if hasattr(self, '_enforce_biological_constraints'):
            df = self._enforce_biological_constraints(df)
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        if not inplace:
            return df
        self.data = df
        return self.data