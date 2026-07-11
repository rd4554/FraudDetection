import sqlite3
import pandas as pd


def create_database():
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL,
        prediction TEXT,
        probability REAL,
        risk TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(amount, prediction, probability, risk):
    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions(amount, prediction, probability, risk)
    VALUES (?, ?, ?, ?)
    """, (amount, prediction, probability, risk))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect("fraud.db")

    history = pd.read_sql_query(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT 20
        """,
        conn
    )

    conn.close()

    return history