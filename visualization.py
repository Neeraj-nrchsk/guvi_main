# main.py
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.data = None
        self._expected_columns = {
            'SepalLengthCm', 'SepalWidthCm', 
            'PetalLengthCm', 'PetalWidthCm', 'Species'
        }

    def load_data(self, source: str = 'local', **kwargs) -> pd.DataFrame:
        if source == 'local':
            self._load_local(**kwargs)
        elif source == 'kaggle':
            self._load_kaggle(**kwargs)
        else:
            raise ValueError(f"Unsupported source: {source}")
        self._validate_dataset()
        return self.data

    def _load_local(self, file_type: str = 'csv', **kwargs) -> None:
        path = Path(self.filepath) if self.filepath else Path('Iris.csv')
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        loaders = {
            'csv': pd.read_csv,
            'excel': pd.read_excel,
            'parquet': pd.read_parquet
        }
        self.data = loaders[file_type](path, **kwargs)

    def _load_kaggle(self, dataset: str = 'uciml/iris', **kwargs) -> None:
        try:
            import kagglehub
            self.data = kagglehub.model_download(dataset, **kwargs)
        except ImportError:
            raise ImportError("kagglehub required. Install with 'pip install kagglehub'")

    def _validate_dataset(self) -> None:
        missing = self._expected_columns - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if self.data.empty:
            raise ValueError("Loaded empty dataset")

class DataIntegrity:
    def __init__(self, data: pd.DataFrame):
        self.original_data = data.copy()
        self.data = data.copy()
        
    def check_quality(self) -> Dict[str, Any]:
        report = {
            'missing': self.data.isna().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'biological': self._biological_checks()
        }
        return report

    def _biological_checks(self) -> Dict[str, bool]:
        valid = {
            'valid_sepal_ratio': (self.data['SepalLengthCm'] / self.data['SepalWidthCm'] < 10).all(),
            'valid_petal_ratio': (self.data['PetalLengthCm'] / self.data['PetalWidthCm'] < 10).all(),
            'positive_measurements': (self.data[['SepalLengthCm', 'SepalWidthCm',
                                               'PetalLengthCm', 'PetalWidthCm']] > 0).all().all()
        }
        return valid

    def clean_data(self) -> pd.DataFrame:
        self.data = self.data.drop_duplicates()
        self._handle_missing()
        return self.data

    def _handle_missing(self) -> None:
        for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
            self.data[col] = self.data[col].fillna(self.data[col].median())

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.encoder = LabelEncoder()
        
    def transform(self) -> pd.DataFrame:
        self._create_features()
        self._encode_target()
        return self.data

    def _create_features(self) -> None:
        self.data['SepalRatio'] = self.data['SepalLengthCm'] / self.data['SepalWidthCm']
        self.data['PetalRatio'] = self.data['PetalLengthCm'] / self.data['PetalWidthCm']
        self.data['SepalArea'] = self.data['SepalLengthCm'] * self.data['SepalWidthCm']
        self.data['PetalArea'] = self.data['PetalLengthCm'] * self.data['PetalWidthCm']

    def _encode_target(self) -> None:
        self.data['SpeciesEncoded'] = self.encoder.fit_transform(self.data['Species'])

class OutlierHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        
    def detect_outliers(self) -> Dict[str, Any]:
        report = {}
        for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = self.data[(self.data[col] < lower) | (self.data[col] > upper)]
            report[col] = {
                'count': len(outliers),
                'indices': outliers.index.tolist()
            }
        return report

    def remove_outliers(self, report: Dict[str, Any]) -> pd.DataFrame:
        outlier_indices = set()
        for col in report:
            outlier_indices.update(report[col]['indices'])
        return self.data.drop(index=list(outlier_indices))

class Visualization:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.data = data.copy()
        self.pca_model = None
        self._numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        self._species_col = self._detect_species_column()
        self._style_config = {
            'palette': 'viridis',
            'figsize': (12, 8),
            'context': 'notebook',
            'font_scale': 1.1
        }

    def _detect_species_column(self) -> Optional[str]:
        for col in self.data.columns:
            if 'species' in col.lower():
                return col
        return None

    def configure_style(self, palette: str = 'viridis', figsize: tuple = (12, 8), context: str = 'notebook', font_scale: float = 1.1) -> None:
        self._style_config.update({
            'palette': palette,
            'figsize': figsize,
            'context': context,
            'font_scale': font_scale
        })
        sns.set(context=context, font_scale=font_scale)

    def plot_distributions(self, save_path: Optional[str] = None) -> None:
        if not self._numeric_cols:
            logger.warning("No numeric columns to plot")
            return
        n_cols = 3
        n_rows = (len(self._numeric_cols) + n_cols - 1) // n_cols
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=self._style_config['figsize'])
            axes = axes.flatten()
            for i, col in enumerate(self._numeric_cols):
                sns.histplot(self.data[col], kde=True, ax=axes[i], color=sns.color_palette()[i%10])
                axes[i].set_title(f'{col} Distribution', fontsize=12)
                axes[i].set_xlabel('')
            for j in range(len(self._numeric_cols), len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            self._handle_plot_output('distributions', save_path)

    def plot_correlations(self, save_path: Optional[str] = None) -> None:
        if len(self._numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return
        corr = self.data[self._numeric_cols].corr()
        with sns.axes_style("white"):
            g = sns.clustermap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                             center=0, dendrogram_ratio=0.1, 
                             cbar_pos=(0.85, 0.3, 0.04, 0.4))
            g.fig.suptitle('Feature Correlation Matrix', y=1.02)
            self._handle_plot_output('correlations', save_path)

    def plot_pairplot(self, sample_size: Optional[int] = 100, save_path: Optional[str] = None) -> None:
        if not self._numeric_cols:
            logger.warning("No numeric columns for pairplot")
            return
        plot_data = self.data.sample(min(sample_size, len(self.data))) if sample_size else self.data
        with sns.plotting_context("notebook", font_scale=1):
            g = sns.pairplot(plot_data, hue=self._species_col, vars=self._numeric_cols, palette=self._style_config['palette'], diag_kind='kde', plot_kws={'alpha': 0.8, 's': 40, 'edgecolor': 'k'}, diag_kws={'linewidth': 0})
            g.fig.suptitle('Feature Pair Relationships', y=1.02)
            self._handle_plot_output('pairplot', save_path)

    def plot_pca_projection(self, n_components: int = 2, save_path: Optional[str] = None) -> Optional[PCA]:
        if len(self._numeric_cols) < 2:
            logger.warning("Not enough numeric columns for PCA")
            return
        X = self.data[self._numeric_cols].fillna(self.data[self._numeric_cols].mean())
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(X)
        with sns.axes_style("whitegrid"):
            plt.figure(figsize=self._style_config['figsize'])
            color_vals = None
            if self._species_col is not None:
                color_vals = self.data[self._species_col].astype('category').cat.codes
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_vals, cmap=self._style_config['palette'], s=70, edgecolor='k', alpha=0.8)
            plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]*100:.1f}%)')
            plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]*100:.1f}%)')
            plt.title('PCA Projection of Iris Features')
            if self._species_col is not None:
                species_labels = self.data[self._species_col].astype('category').cat.categories
                handles, _ = scatter.legend_elements()
                legend = plt.legend(handles=handles, labels=species_labels,
                                  title='Species', bbox_to_anchor=(1.05, 1), 
                                  loc='upper left')
                plt.gca().add_artist(legend)
            plt.tight_layout()
            self._handle_plot_output('pca_projection', save_path)
            return self.pca_model

    def plot_outlier_analysis(self, save_path: Optional[str] = None) -> None:
        if not self._numeric_cols:
            logger.warning("No numeric columns for outlier analysis")
            return
        with sns.axes_style("whitegrid"):
            plt.figure(figsize=self._style_config['figsize'])
            melt_df = self.data.melt(value_vars=self._numeric_cols)
            ax = sns.boxplot(x='variable', y='value', data=melt_df,
                           palette='Set2', width=0.6, legend=False)
            sns.swarmplot(x='variable', y='value', data=melt_df, color='black', alpha=0.5, size=3, ax=ax)
            plt.xticks(rotation=45)
            plt.title('Outlier Distribution Analysis')
            plt.xlabel('Features')
            plt.ylabel('Values')
            self._handle_plot_output('outlier_analysis', save_path)

    def _handle_plot_output(self, plot_name: str, save_path: Optional[str] = None) -> None:
        if save_path:
            full_path = f"{save_path}_{plot_name}.png"
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved {plot_name} plot to {full_path}")
        plt.show()
        plt.close()

    def plot_all(self, save_dir: Optional[str] = None) -> None:
        plots = [
            self.plot_distributions,
            self.plot_correlations,
            self.plot_pairplot,
            self.plot_pca_projection,
            self.plot_outlier_analysis
        ]
        for plot_func in plots:
            try:
                plot_func(save_path=save_dir)
            except Exception as e:
                logger.error(f"Failed to generate {plot_func.__name__}: {str(e)}")
                
    def plot_key_findings(self, save_dir: Optional[str] = None) -> None:
        """Alias for plot_all to match main.py usage."""
        self.plot_all(save_dir=save_dir)