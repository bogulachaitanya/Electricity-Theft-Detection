import plotly.express as px
import streamlit as st
import numpy as np

def render_risk_hierarchy(df):
    """
    Renders a Treemap representing the hierarchy of risk.
    """
    if df is None or df.empty:
        st.warning("No data for risk map.")
        return
    
    try:
        plot_df = df.copy()
        if 'Risk_Tier' not in plot_df.columns:
            return

        if 'FEEDER_ID' not in plot_df.columns:
            plot_df['FEEDER_ID'] = 'Line_' + (np.arange(len(plot_df)) % 5 + 1).astype(str)
            plot_df['TRANSFORMER_ID'] = 'TX_' + (np.arange(len(plot_df)) % 15 + 1).astype(str)

        fig = px.treemap(plot_df, 
                         path=['FEEDER_ID', 'TRANSFORMER_ID', 'Risk_Tier'], 
                         values='Final_Risk_Score',
                         color='Final_Risk_Score',
                         color_continuous_scale='RdYlGn_r',
                         title="Network Risk Hierarchy")
        
        fig.update_layout(template='plotly_dark', margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render risk hierarchy: {e}")

def render_geographic_map(df):
    """
    Renders a geographic map showing household locations and theft status.
    If lat/lon are missing, it simulates them for demonstration.
    """
    if df is None or df.empty:
        st.warning("No data for geographic map.")
        return

    try:
        plot_df = df.copy()
        
        # Check for lat/long, simulate if missing
        if 'latitude' not in plot_df.columns or 'longitude' not in plot_df.columns:
            # Center near a dummy location (e.g. London area for demo, or Zhengzhou for SGCC)
            # Let's use Zhengzhou, China as it's common for SGCC data
            base_lat, base_lon = 34.7466, 113.6253 
            np.random.seed(42)  # For consistent simulation
            plot_df['latitude'] = base_lat + np.random.normal(0, 0.05, size=len(plot_df))
            plot_df['longitude'] = base_lon + np.random.normal(0, 0.05, size=len(plot_df))
            st.sidebar.info("🌐 Geographic coordinates simulated for visualization.")

        tier_colors = {
            "Normal": "#4CAF50",
            "Suspicious": "#FFC107",
            "High Risk": "#FF9800",
            "Theft": "#F44336"
        }

        fig = px.scatter_mapbox(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="Risk_Tier",
            size="Final_Risk_Score",
            hover_name="CONS_NO" if "CONS_NO" in plot_df.columns else None,
            hover_data=["Final_Risk_Score", "Risk_Tier"],
            color_discrete_map=tier_colors,
            zoom=10,
            mapbox_style="carto-darkmatter",
            title="Geographic Risk Distribution & Theft Locations"
        )

        fig.update_layout(
            template='plotly_dark',
            margin={"t": 50, "l": 10, "r": 10, "b": 10},
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01}
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render geographic map: {e}")

