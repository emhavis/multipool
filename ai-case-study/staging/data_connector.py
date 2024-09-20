import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.util import deprecations
from sqlalchemy.sql import text
deprecations.SILENCE_UBER_WARNING = True
# create database connection
import psycopg2
import psycopg2.extras
from typing import Iterator, Dict, Any

def insert_execute_batch(connection, ssql, df_dict) -> None:
    
    connection = psycopg2.connect(
        host="128.1.227.35",
        port='5432',
        database="medols",
        user="ai_gpu",
        password='42DRGnrfdnV1',
    )

    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            psycopg2.extras.execute_batch(cursor, ssql, df_dict)

    except:
        # Rollback changes if there's an error
        print(f"Error executing query: {e}")

    finally:
        connection.close()

def execute_query_psql(query, params=None):
    # Set your PostgreSQL connection parameters
    db_params = {
        'host': '128.1.227.35',
        'port': '5432',
        'database': 'medols',
        'user': 'postgres',
        'password': 'FEWcTB3JIX5gK4T06c1MdkM9N2S8w9pb',
    }

    # Create a SQLAlchemy engine
    engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

    # Create a metadata object
    metadata = MetaData()

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Execute the query with optional parameters
        result = session.execute(text(query), params)

        # Check if the query is a SELECT query
        is_select_query = result.returns_rows

        if is_select_query:
            # Fetch the data and return as a Pandas DataFrame
            columns = result.keys()
            fetched_data = result.fetchall()
            df = pd.DataFrame(fetched_data, columns=columns)
            # print("Fetched Data as DataFrame:")
            # print(df)
            return df
        else:
            # Get the number of rows affected for non-SELECT queries
            rows_affected = result.rowcount

            # Commit the changes to the database for non-SELECT queries
            session.commit()

            return rows_affected
    except Exception as e:
        # Rollback changes if there's an error
        session.rollback()
        print(f"Error executing query: {e}")
    finally:
        # Close the session
        session.close()




