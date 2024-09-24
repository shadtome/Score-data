import sqlite3
import pandas as pd
import os

def csvs_to_sql(csv_files, path, name):
    """ This is used to take the collection of csv files and put it together 
    as a database file for use in sqlite3"""
    sql_file = os.path.join(path,f'{name}.db')
    con = sqlite3.connect(sql_file)

    for csv in csv_files:
        df = pd.read_csv(os.path.join(path,csv))

        table_name = csv.split('.')[0]
        df.to_sql(table_name,con,if_exists='replace',index=False)

    con.commit()
    con.close()