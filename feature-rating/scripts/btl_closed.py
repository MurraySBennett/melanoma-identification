import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

plt.style.use("ggplot")

df = pd.read_csv(os.path.join(os.getcwd(), "..", "data", "closedData.csv"))
df = df.head(200)


def get_estimate(i, p, df):
    get_prob = lambda i, j: np.nan if i ==j else p.iloc[i] + p.iloc[j]
    n = df.iloc[i].sum()

    d_n = df.iloc[i] + df.iloc[:, i]
    d_d = pd.Series([get_prob(i, j) for j in range(len(p))], index=p.index)
    d = (d_n / d_d).sum()
    return n / d


def estimate_p(p, df):
    return pd.Series([get_estimate(i, p, df) for i in range(df.shape[0])], index=p.index)


def iterate(df, p=None, n=20, sorted=True):
    if p is None:
        p = pd.Series([1 for _ in range(df.shape[0])], index=list(df.columns))
    estimates = [p]
    for _ in range(n):
        p = estimate_p(p, df)
        p = p / p.sum()
        estimates.append(p)

    p = p.sort_values(ascending=False) if sorted else p
    return p, pd.DataFrame(estimates)


def get_winner(r):
    if r.responselist == 0:
        return r.leftlist
    else:
        return r.rightlist


def get_loser(r):
    if r.responselist == 0:
        return r.rightlist
    else:
        return r.leftlist


df['winner'] = df.apply(get_winner, axis=1)
df['loser'] = df.apply(get_loser, axis=1)

w = df.winner.value_counts().sort_index()
l = df.loser.value_counts().sort_index()

wl_df = pd.DataFrame([w, l]).T.rename(columns={"winner": "wins", "loser": "losses"})
wl_df['n'] = wl_df.wins + wl_df.losses

# print(wl_df.head())

# _ = wl_df[['wins', 'losses']].plot(kind='bar', figsize=(15, 4), title='Closed Wins/Losses')
# plt.show()

images = sorted(list(set(df.leftlist) | set(df.rightlist)))

# img2i = {img: i for i, img in enumerate(images)}
#
# df = df\
#     .groupby(['winner', 'loser'])\
#     .agg('count')\
#     .drop(columns=['rightlist', 'responselist'])\
#     .rename(columns={'leftlist': 'n'})\
#     .reset_index()
#
# df['r'] = df['winner'].apply(lambda img: img2i[img])
# df['c'] = df['loser'].apply(lambda img: img2i[img])
#
# n_images = len(images)
# mat = np.zeros([n_images, n_images])
#
# for _, r in df.iterrows():
#     mat[r.r, r.c] = r.n
#
# df = pd.DataFrame(mat, columns=images, index=images)
# print(df.head())

# p, estimates = iterate(df, n=100)



# fig, ax = plt.subplots(figsize=(10,10))
# ax.axis("off")
# ax.set_xticks([])
# ax.set_yticks([])
# sns.heatmap(df)
# plt.show()


# Regression format
def get_vector(r):
    # y = {'y': 1 if r.h_score > r.a_score else 0}
    y = {'y': r.responselist}
    v = {img: 0 for img in images}
    v[r.leftlist] = -1
    v[r.rightlist] = 1
    return {**y, **v}


X = pd.DataFrame(list(df.apply(get_vector, axis=1)))
y = X.y
X = X[[c for c in X.columns if c != 'y']]

l1_model = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=True)
l1_model.fit(X, y)
q = sorted(list(zip(X.columns, l1_model.coef_[0])), key=lambda tup: tup[1], reverse=True)
q = pd.Series([c for _, c in q], index=[t for t, _ in q])


l2_model = LogisticRegression(penalty='l2', solver='liblinear', fit_intercept=True)
l2_model.fit(X, y)
r = sorted(list(zip(X.columns, l2_model.coef_[0])), key=lambda tup: tup[1], reverse=True)
r = pd.Series([c for _, c in r], index=[t for t, _ in r])

rank_df = pd.DataFrame([q, r]).T.rename(columns={0: 'r', 1: 'q'})

print(rank_df.corr())