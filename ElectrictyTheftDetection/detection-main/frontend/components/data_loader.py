import pandas as pd
import sqlite3
import os
import streamlit as st
import numpy as np

# Resolve paths relative to project root (one level up from frontend/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class DashboardDataLoader:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(PROJECT_ROOT, "data", "investigations.db")

    @st.cache_data
    def load_risk_csv(_self, file_path=None):
        """Loads the main risk score CSV with caching."""
        if file_path is None:
            file_path = os.path.join(PROJECT_ROOT, "data", "processed", "final_risk_scores.csv")
        if not os.path.exists(file_path):
            return None
        return pd.read_csv(file_path)

    @st.cache_data
    def load_raw_series(_self, cons_no, file_path=None):
        """Loads full time-series for a specific consumer."""
        if file_path is None:
            file_path = os.path.join(PROJECT_ROOT, "data", "raw", "sgcc", "data set.csv")
        if not os.path.exists(file_path):
            return None
            
        # Optimization: only read header first to get column index or use chunks
        # For simplicity in demo, we read row
        df = pd.read_csv(file_path)
        user_row = df[df['CONS_NO'] == cons_no]
        
        if user_row.empty:
            return None
            
        # Extract date columns
        exclude = ['CONS_NO', 'FLAG', 'ID']
        dates = [c for c in df.columns if c not in exclude]
        
        # Values vertical
        values = user_row[dates].values.flatten()
        return pd.DataFrame({'Day': np.arange(len(dates)), 'Consumption': values})

    def get_db_connection(self):
        """Returns a connection to the SQLite local database."""
        if not os.path.exists(self.db_path):
            return None
        return sqlite3.connect(self.db_path)

    def fetch_investigation_stats(self):
        """Fetches counts of Open vs Resolved cases from SQL."""
        try:
            conn = self.get_db_connection()
            if conn is None:
                return pd.DataFrame({'status': ['In Progress', 'Completed'], 'count': [2, 1]})
                
            query = "SELECT status, COUNT(*) as count FROM investigations GROUP BY status"
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except:
            # Return empty skeleton if DB not initialized
            return pd.DataFrame({'status': ['In Progress', 'Completed'], 'count': [2, 1]})
