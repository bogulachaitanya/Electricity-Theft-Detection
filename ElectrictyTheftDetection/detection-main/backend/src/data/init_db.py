import sqlite3
import pandas as pd
import os

def init_db():
    db_path = 'data/investigations.db'
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create investigations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS investigations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        consumer_id TEXT,
        inspector_name TEXT,
        status TEXT,
        finding TEXT,
        date TEXT
    )
    ''')
    
    # Check if we have some data to seed
    try:
        if os.path.exists('data/processed/final_risk_scores.csv'):
            scores_df = pd.read_csv('data/processed/final_risk_scores.csv')
            # Seed with some high risk cases
            high_risk = scores_df[scores_df['Risk_Tier'] == 'Theft'].head(5)
            
            for _, row in high_risk.iterrows():
                cursor.execute('''
                INSERT INTO investigations (consumer_id, inspector_name, status, finding, date)
                VALUES (?, ?, ?, ?, ?)
                ''', (row['CONS_NO'], 'Auto-System', 'Completed', 'Theft Confirmed', '2024-03-20'))
            
            # Seed some pending ones
            suspicious = scores_df[scores_df['Risk_Tier'] == 'High Risk'].head(3)
            for _, row in suspicious.iterrows():
                cursor.execute('''
                INSERT INTO investigations (consumer_id, inspector_name, status, finding, date)
                VALUES (?, ?, ?, ?, ?)
                ''', (row['CONS_NO'], 'Auto-System', 'In Progress', 'Field Check Pending', '2024-03-21'))
        else:
            print("final_risk_scores.csv not found. Database created but not seeded.")
    except Exception as e:
        print(f"Error seeding database: {e}")
        
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_db()
