import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, confusion_matrix
import numpy as np

def plot_roc_curve(y_true, y_probs, model_name="Model"):
    """Generates an interactive Plotly ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve', mode='lines', line=dict(color='firebrick', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess', mode='lines', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark'
    )
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Generates an interactive heatmap for the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Theft'],
                    y=['Normal', 'Theft'],
                    text_auto=True,
                    color_continuous_scale='RdBu_r')
    
    fig.update_layout(title="Confusion Matrix Heatmap", template='plotly_dark')
    return fig

def plot_consumption_anomaly(dates, values, anomalies=None, title="Consumption Pattern"):
    """Plots time-series with highlighted anomaly points."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, name='Consumption', mode='lines', line=dict(color='#4CAF50')))
    
    if anomalies is not None:
        anomaly_dates = [dates[i] for i in range(len(anomalies)) if anomalies[i] == 1]
        anomaly_values = [values[i] for i in range(len(anomalies)) if anomalies[i] == 1]
        fig.add_trace(go.Scatter(x=anomaly_dates, y=anomaly_values, name='Anomaly Detected', 
                                mode='markers', marker=dict(color='red', size=8)))
        
    fig.update_layout(title=title, template='plotly_dark', xaxis_title="Date", yaxis_title="kWh")
    return fig
