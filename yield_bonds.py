import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pymc3 as pm  # Pour une approche bayésienne
import seaborn as sns

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


pca = PCA(n_components=df_yld.shape[1])
pca.fit(df_yld)
'''explained_variance = pd.Series([i*100 for i in pca.explained_variance_ratio_], index=[f'PC{i+1}' for i in range(len(df_yld.columns))])
fig = px.bar(explained_variance)
fig.update_layout(showlegend=False)
#fig.show()

fig =px.bar(np.cumsum(explained_variance))
fig.show()'''

yield_pca = pca.transform(df_yld)
yield_pca_df = pd.DataFrame(yield_pca[:,:3], index= df_yld.index, columns=['PC1', 'PC2', 'PC3'])
#fig = px.line(yield_pca_df)


'''pc1 = np.array(yield_pca_df['PC1'])
dt = 1  # Pas de temps (par exemple, jours ou mois)

# Différences successives
X_t = pc1[:-1]
X_t1 = pc1[1:]
delta_X = X_t1 - X_t

# Régression linéaire pour estimer les paramètres
reg = LinearRegression().fit(X_t.reshape(-1, 1), delta_X)
theta = -reg.coef_[0] / dt
mu = reg.intercept_ / (theta * dt)

# Estimation de sigma
sigma = np.sqrt(np.var(delta_X - reg.predict(X_t.reshape(-1, 1))) / dt)

print(f"Paramètres estimés : θ={theta:.4f}, μ={mu:.4f}, σ={sigma:.4f}")

# Simulation de trajectoires
n_steps = 50  # Nombre de pas de simulation
n_scenarios = 10000  # Nombre de trajectoires simulées
X_sim = np.zeros((n_steps, n_scenarios))
X_sim[0, :] = pc1[-1]  # Dernière valeur observée comme valeur initiale

for t in range(1, n_steps):
    dW = np.random.normal(0, 1, n_scenarios)
    X_sim[t, :] = X_sim[t - 1, :] + theta * (mu - X_sim[t - 1, :]) * dt + sigma * np.sqrt(dt) * dW

# Moyenne et intervalle de confiance
mean_X = np.mean(X_sim, axis=1)
std_X = np.std(X_sim, axis=1)
lower_bound = mean_X - std_X
upper_bound = mean_X + std_X

# Combiner données historiques et prévisions
time_historical = np.arange(len(pc1))
time_forecast = np.arange(len(pc1), len(pc1) + n_steps)

# Visualisation
plt.figure(figsize=(12, 7))

# Données historiques
plt.plot(time_historical, pc1, label="Données historiques", color="black", linewidth=2)

# Prévisions moyennes
plt.plot(time_forecast, mean_X, label="Prévision moyenne", color="blue", linestyle="--", linewidth=2)

# Intervalle de confiance
plt.fill_between(time_forecast, lower_bound, upper_bound, color="blue", alpha=0.3, label="Intervalle 1σ (prédictions)")

# Options du graphique
plt.title("Données historiques et prévisions avec modèle OU", fontsize=16)
plt.xlabel("Temps", fontsize=14)
plt.ylabel("PC1", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# Données historiques
pc1 = np.array(yield_pca_df['PC1'])
dt = 1  # Pas de temps (par exemple, jours ou mois)


# Estimation des paramètres avec le maximum de vraisemblance
def negative_log_likelihood(params, X, dt):
    theta, mu, sigma = params
    X_t = X[:-1]
    X_t1 = X[1:]
    mu_t = X_t + theta * (mu - X_t) * dt
    var_t = sigma ** 2 * dt
    likelihood = -0.5 * np.sum(np.log(2 * np.pi * var_t) + ((X_t1 - mu_t) ** 2) / var_t)
    return -likelihood


# Estimation initiale des paramètres
initial_guess = [0.1, np.mean(pc1), np.std(pc1)]
bounds = [(0, None), (None, None), (0, None)]
result = minimize(negative_log_likelihood, initial_guess, args=(pc1, dt), bounds=bounds)
theta, mu, sigma = result.x

print(f"Paramètres estimés (MLE): θ={theta:.4f}, μ={mu:.4f}, σ={sigma:.4f}")

# Simulation de trajectoires
n_steps = 50  # Nombre de pas de simulation
n_scenarios = 5000  # Nombre de trajectoires simulées
X_sim = np.zeros((n_steps, n_scenarios))
X_sim[0, :] = pc1[-1]  # Dernière valeur observée comme valeur initiale

for t in range(1, n_steps):
    dW = np.random.normal(0, 1, n_scenarios)
    X_sim[t, :] = X_sim[t - 1, :] + theta * (mu - X_sim[t - 1, :]) * dt + sigma * np.sqrt(dt) * dW

# Moyenne et intervalle de confiance
mean_X = np.mean(X_sim, axis=1)
std_X = np.std(X_sim, axis=1)
lower_bound = mean_X - 1 * std_X  # Intervalle 95%
upper_bound = mean_X + 1 * std_X

# Visualisation
time_historical = np.arange(len(pc1))
time_forecast = np.arange(len(pc1), len(pc1) + n_steps)

plt.figure(figsize=(14, 8))

# Données historiques
plt.plot(time_historical, pc1, label="Données historiques", color="black", linewidth=2)

# Prévisions moyennes
plt.plot(time_forecast, mean_X, label="Prévision moyenne", color="blue", linestyle="--", linewidth=2)

# Intervalle de confiance
plt.fill_between(time_forecast, lower_bound, upper_bound, color="blue", alpha=0.3, label="Intervalle 95% (prédictions)")

# Options du graphique
plt.title("Données historiques et prévisions avec modèle OU amélioré", fontsize=16)
plt.xlabel("Temps", fontsize=14)
plt.ylabel("PC1", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Étape supplémentaire : Validation bayésienne ---- #
print("\nApproche bayésienne avec PyMC3...")
with pm.Model() as model:
    theta_b = pm.HalfNormal("theta", sigma=1)
    mu_b = pm.Normal("mu", mu=np.mean(pc1), sigma=np.std(pc1))
    sigma_b = pm.HalfNormal("sigma", sigma=1)

    X_obs = pc1[:-1]
    X_next = pc1[1:]
    mean_next = X_obs + theta_b * (mu_b - X_obs) * dt
    likelihood = pm.Normal("likelihood", mu=mean_next, sigma=sigma_b * np.sqrt(dt), observed=X_next)

    trace = pm.sample(2000, return_inferencedata=True, progressbar=True)

pm.plot_trace(trace)
plt.tight_layout()
plt.show()









'''#fig.show()
# 
# 
# 

coefs = pd.DataFrame(pca.components_, index=[f'PC{i+1}' for i in range(len(df_yld.columns))])
coefs = coefs.iloc[:3]

fig = px.bar(coefs.T, barmode = 'group')
#fig.show()'''


'''pca = PCA(n_components=3)
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
#fig.show()'''

'''windows_size = 20
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
#fig.show()'''



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




#apply_PCA(df_yld, df_yld.index[50], 20)
