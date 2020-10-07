import pandas as pd 

#%%
def clean_cols(df, fill_int = 0, fill_str = ''):
    for k in df.keys():
        filling = fill_int
        df[k] = df[k].fillna(filling)
    return(df)

def merge_based_on_first(df1, df2):
    """ Merge two dataframes converting the types of the second df to
    that of the first if needed and only keeping column names appearing
    in either of the two dfs"""
    com_cols = df1[df1.columns & df2.columns].keys()
    df1_types = df1[com_cols].dtypes
    df2[com_cols] = df2[com_cols].astype(df1_types)
    df = pd.concat([df1,df2],ignore_index=True)
    return(df)

def commas2points2float(df, cols):
    df[cols] = df[cols].apply(lambda x: x.str.replace(',',''))
    df[cols] = df[cols].apply(lambda x: x.str.replace('%',''))
    df[cols] = df[cols].apply(lambda x: x.astype('float'))
    return(df)

def ts2dateString(df, cols):
    df[cols] = pd.to_datetime(df[cols]).dt.strftime('%Y-%m-%d')
    return(df)
# %%
