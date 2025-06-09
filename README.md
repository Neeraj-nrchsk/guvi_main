# Iris Data Analysis Project

This project provides a comprehensive pipeline for analyzing the classic Iris dataset. It includes modules for data loading, cleaning, feature engineering, statistical summary, pattern analysis, outlier detection, and both static and interactive visualizations.

## Features
- **Data Loading**: Load data from local files, Kaggle, or URLs with automatic validation
- **Data Integrity**: Assess and clean data for missing values, duplicates, and biological plausibility
- **Feature Engineering**: Create biologically meaningful features (ratios, areas) and scale them for analysis
- **Summary Statistics**: Generate detailed statistical summaries and domain-specific insights
- **Pattern Analysis**: Identify patterns, correlations, anomalies, and feature importance using multiple algorithms
- **Outlier Handling**: Detect and handle outliers using robust statistical methods (IQR, Z-score, Isolation Forest)
- **Static Visualizations**: Create publication-ready plots using Matplotlib and Seaborn
- **Interactive Visualizations**: Generate interactive HTML visualizations using Plotly

## File Structure
### Core Pipeline
- `main.py`: Complete analysis pipeline demonstrating all modules
- `data_loader.py`: Data loading utilities with validation
- `data_integrity.py`: Data quality checks and cleaning procedures
- `feature_engineering.py`: Feature creation, transformation, and scaling
- `summary_stats.py`: Statistical summaries and domain insights
- `pattern_analysis.py`: Pattern detection, correlation analysis, and anomaly detection
- `outlier_handling.py`: Multi-method outlier detection and handling
- `visualization.py`: Static visualization suite (Matplotlib/Seaborn)
- `iris_visualizations.py`: Interactive visualization suite (Plotly)

### Data and Output
- `Iris.csv`: Classic Iris dataset
- `*.html`: Generated interactive visualization files
- `__pycache__/`: Python compiled bytecode files

## Requirements
See `requirements.txt` for dependencies. Install with:

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis Pipeline
Execute the main pipeline that demonstrates all modules:
```bash
python main.py
```

### Generate Interactive Visualizations
Create interactive HTML visualizations:
```bash
python iris_visualizations.py
```

### Individual Module Usage
You can also use individual modules for custom analysis:

```python
from data_loader import DataLoader
from visualization import Visualization

# Load data
loader = DataLoader("Iris.csv")
data = loader.load_data()

# Create visualizations
viz = Visualization(data)
viz.plot_all()
```

## Generated Outputs

### Interactive HTML Visualizations
- `parallel_coordinates.html`: Multi-dimensional feature relationships
- `3d_scatter.html`: 3D scatter plot in feature space
- `pca.html`: Principal Component Analysis projection
- `pair_plot.html`: Feature pair relationships matrix
- `radar.html`: Species feature distribution radar chart
- `box_swarm.html`: Distribution analysis with outliers
- `ratio_comparison.html`: Sepal vs Petal ratio analysis

### Static Visualizations
The main pipeline generates static plots including:
- Feature distributions and correlations
- PCA projections with species clustering
- Outlier analysis and detection results
- Statistical summary visualizations

## Key Features Implemented

### Data Analysis
- **Automated data validation** with biological plausibility checks
- **Feature engineering** including ratios (Sepal/Petal length-to-width) and areas
- **Multi-method outlier detection** (IQR, Z-score, Isolation Forest)
- **Pattern analysis** with correlation matrices and feature importance
- **Statistical summaries** with domain-specific insights

### Visualizations
- **Static plots** using Matplotlib/Seaborn for publication-ready figures
- **Interactive plots** using Plotly for exploratory data analysis
- **3D visualizations** for multi-dimensional feature relationships
- **PCA projections** for dimensionality reduction analysis

## Notes
- The project is modular and can be extended to other tabular datasets with similar structure.
- Some features (e.g., Kaggle download) require additional setup (e.g., Kaggle API credentials).

## License
MIT License
