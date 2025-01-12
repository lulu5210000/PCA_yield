import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from sklearn.metrics import explained_variance_score


def read_data(filename: str, yd: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep='\t')
    df.set_index('date', inplace=True)
    df.rename(columns={'clot': yd + 'Y'}, inplace=True)
    df = df[[yd + 'Y']]
    return df

def merge_data(df_list: list) -> pd.DataFrame:
    df = pd.concat(df_list, axis=1)
    return df

list_yield = [None] * 12
list_name_yield = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '20']

for i in range(len(list_yield)):
    list_yield[i] = read_data('ITABENCHMARK' + list_name_yield[i] + 'A_2025-01-12.txt', list_name_yield[i])

df_yld = merge_data(list_yield)
df_yld = df_yld.dropna()
df_yld.index = pd.to_datetime(df_yld.index, dayfirst=True)
df_yld = df_yld[:]
def print_chart() -> None:
    plt.figure(figsize=(12, 6))
    for column in df_yld.columns:
        plt.plot(df_yld[column], label=column)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Yield")
    plt.title("Évolution des Yields")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


'''pca = PCA(n_components=df_yld.shape[1])
pca.fit(df_yld)
explained_variance = pd.Series([i*100 for i in pca.explained_variance_ratio_], index=[f'PC{i+1}' for i in range(len(df_yld.columns))])
fig = px.bar(explained_variance)
fig.update_layout(showlegend=False)
#fig.show()

fig =px.bar(np.cumsum(explained_variance))
fig.show()

yield_pca = pca.transform(df_yld)
yield_pca_df = pd.DataFrame(yield_pca[:,:3], index= df_yld.index, columns=['PC1', 'PC2', 'PC3'])
fig = px.line(yield_pca_df)
#fig.show()

coefs = pd.DataFrame(pca.components_, index=[f'PC{i+1}' for i in range(len(df_yld.columns))])
coefs = coefs.iloc[:3]

fig = px.bar(coefs.T, barmode = 'group')
#fig.show()'''


pca = PCA(n_components=3)
pca.fit(df_yld)

reconstructed_df = pd.DataFrame(pca.inverse_transform(pca.transform(df_yld)), index=df_yld.index, columns=df_yld.columns)
error = df_yld - reconstructed_df
plt.plot(df_yld.iloc[0,:],label='Yield Curve')
plt.plot(reconstructed_df.iloc[0,:],label='Yield Curve Reconstructed')
plt.legend()
#plt.show()

fig = px.line(error)
#plt.show()

error_zscore = (error - error.mean()) / error.std()
fig = px.line(error_zscore)
#plt.show()

fig = px.bar(error_zscore.iloc[-4])
fig.add_hrect(y0=0, y1=1, line_width=0, fillcolor='red', opacity=0.2)
fig.add_hrect(y0=-2.3, y1=0, line_width=0, fillcolor='green', opacity=0.2)
#fig.show()

windows_size = 20
rolling_pcs = {}

for start in range(len(df_yld) - windows_size + 1):
    windows_data = df_yld[start:start + windows_size]
    pca_window = PCA(n_components=3)
    pca_window.fit(windows_data)
    rolling_pcs[windows_data.index[0]] = list(pca_window.explained_variance_ratio_)

rolling_pcs = pd.DataFrame(rolling_pcs, index = ['PC1', 'PC2', 'PC3']).T

fig = px.line(rolling_pcs)
#fig.show()

fig = px.line(rolling_pcs.cumsum(axis=1))
#fig.show()



def apply_PCA(df,start, window):
    id_start = df.index.get_loc(start)

    window_data = df.iloc[id_start: id_start + window]
    pca_window = PCA(n_components=3)
    pca_window.fit(window_data)


    explained_variance = pd.Series([i*100 for i in pca_window.explained_variance_ratio_])
    fig = px.bar(explained_variance)
    fig.update_layout(showlegend=False)
    fig.show()

    coefs = pd.DataFrame(pca_window.components_, columns=df.columns, index=['PC1', 'PC2', 'PC3'])
    fig = px.bar(coefs.iloc[:3].T, barmode='group')
    fig.show()

    yield_pca = pca.transform(window_data)
    yield_pcf_df = pd.DataFrame(yield_pca[:,:3], index = windows_data.index, columns=['PC1', 'PC2', 'PC3'])
    fig = px.line(yield_pcf_df)
    fig.show()


apply_PCA(df_yld, df_yld.index[50], 20)
