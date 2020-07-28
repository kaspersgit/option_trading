import pandas as pd 
from datetime import datetime

def get_mature_df(df, symbolType=['Call']):
    # Select only mature cases (and exclude options with less then 5 days to expiration)
    df_select = df[pd.to_datetime(df['expirationDate']) < datetime.today()]
    df_select = df_select[df_select['daysToExpiration'] > 4]
    df_select = df_select[df_select['symbolType'].isin(symbolType)]
    return(df_select)