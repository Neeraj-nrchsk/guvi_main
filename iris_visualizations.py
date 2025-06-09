import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader

# Initialize data loader and load dataset
loader = DataLoader("Iris.csv")
iris_df = loader.load_data()

# Feature engineering
iris_df['SepalRatio'] = iris_df['SepalLengthCm'] / iris_df['SepalWidthCm']
iris_df['PetalRatio'] = iris_df['PetalLengthCm'] / iris_df['PetalWidthCm']
iris_df['SepalArea'] = iris_df['SepalLengthCm'] * iris_df['SepalWidthCm']
iris_df['PetalArea'] = iris_df['PetalLengthCm'] * iris_df['PetalWidthCm']

# Create interactive visualizations
def create_visualizations(df):
    # 1. Parallel Coordinates Plot
    # Convert species to numeric for parallel coordinates
    df_parallel = df.copy()
    df_parallel['Species_numeric'] = pd.Categorical(df_parallel['Species']).codes
    
    parallel_fig = px.parallel_coordinates(
        df_parallel,
        color="Species_numeric",
        dimensions=['SepalLengthCm', 'SepalWidthCm', 
                   'PetalLengthCm', 'PetalWidthCm'],
        title="Parallel Coordinates Analysis of Iris Features by Species",
        color_continuous_scale=px.colors.qualitative.Set1
    )
    parallel_fig.update_layout(
        height=600,
        hovermode='closest',
        font=dict(family="Arial", size=12)
    )
    
    # 2. 3D Scatter Plot
    scatter_3d = px.scatter_3d(
        df,
        x='PetalLengthCm',
        y='PetalWidthCm',
        z='SepalLengthCm',
        color='Species',
        size='SepalWidthCm',
        symbol='Species',
        opacity=0.8,
        title="3D Feature Space Visualization",
        hover_name='Species',
        hover_data=['SepalRatio', 'PetalRatio']
    )
    scatter_3d.update_layout(
        scene=dict(
            xaxis_title='Petal Length (cm)',
            yaxis_title='Petal Width (cm)',
            zaxis_title='Sepal Length (cm)'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    # 3. PCA Projection
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = df[features].values
    X = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=['PC1', 'PC2'])
    pca_df['Species'] = df['Species'].values
    
    pca_fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Species',
        title=f"PCA Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)",
        hover_data={
            'PC1': ':.3f',
            'PC2': ':.3f',
            'Species': True
        }
    )
    pca_fig.update_traces(marker=dict(size=12, opacity=0.8))
    pca_fig.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )
    
    # 4. Interactive Pair Plot
    pair_fig = px.scatter_matrix(
        df,
        dimensions=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
        color='Species',
        symbol='Species',
        title="Feature Pair Relationships",
        hover_name='Species',
        height=800
    )
    pair_fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    # 5. Feature Distribution Radar Chart
    agg_df = df.groupby('Species').mean().reset_index()
    
    radar_fig = go.Figure()
    
    species = agg_df['Species'].unique()
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    for specie in species:
        radar_fig.add_trace(go.Scatterpolar(
            r=agg_df[agg_df['Species'] == specie][features].values[0],
            theta=features,
            fill='toself',
            name=specie
        ))
    
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 8])),
        showlegend=True,
        title="Feature Distribution Radar Chart by Species",
        height=500
    )
    
    # 6. Box Plot with Swarm
    box_fig = make_subplots(rows=2, cols=2)
    
    features = [
        ('SepalLengthCm', 1, 1),
        ('SepalWidthCm', 1, 2),
        ('PetalLengthCm', 2, 1),
        ('PetalWidthCm', 2, 2)
    ]
    
    for feature, row, col in features:
        box_fig.add_trace(
            go.Box(
                y=df[feature],
                x=df['Species'],
                name=feature,
                boxpoints='all',
                jitter=0.5,
                pointpos=0,
                marker=dict(size=4)
            ),
            row=row, col=col
        )
    
    box_fig.update_layout(
        title_text="Distribution Analysis with Outliers",
        height=600,
        showlegend=False
    )
    
    # 7. Ratio Comparison
    ratio_fig = px.scatter(
        df,
        x='SepalRatio',
        y='PetalRatio',
        color='Species',
        size='PetalArea',
        hover_name='Species',
        title="Sepal vs Petal Ratio Comparison",
        trendline='ols',
        marginal_x='box',
        marginal_y='violin'
    )
    ratio_fig.update_layout(height=600)
    
    return {
        "parallel_coordinates": parallel_fig,
        "3d_scatter": scatter_3d,
        "pca": pca_fig,
        "pair_plot": pair_fig,
        "radar": radar_fig,
        "box_swarm": box_fig,
        "ratio_comparison": ratio_fig
    }

# Generate all visualizations
visualizations = create_visualizations(iris_df)

# Save visualizations as HTML files
for name, fig in visualizations.items():
    fig.write_html(f"{name}.html")