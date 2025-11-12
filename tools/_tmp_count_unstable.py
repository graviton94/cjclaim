import pandas as pd
p='artifacts/stability_report.csv'
df=pd.read_csv(p)
unstable=df[~df['stable'].astype(bool) & df['stable'].notnull()]
print('total_checked=',len(df))
print('unstable_count=',len(unstable))
print(unstable[['series_id','n_train','max_abs_arroot','stable']].head(10).to_string(index=False))
