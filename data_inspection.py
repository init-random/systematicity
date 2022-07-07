from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from main import ApplicationData
import numpy as np
import pandas as pd


data_fn = 'data/data.csv'
data = ApplicationData(data_fn)
demographic_data = pd.concat((data.demographic_data, data.demographic_data_tst))
y = np.hstack((data.y_trn, data.y_tst))

tsne = TSNE()
d = tsne.fit_transform(demographic_data)
db = DBSCAN(eps=2.65)
db.fit(d)
colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow']
print(db.labels_.max())
for color, l in zip(colors, range(db.labels_.max())):
    idx = np.where(db.labels_ == l)[0]
    plt.scatter(d[idx, 0], d[idx, 1], color=color, alpha=0.2)
plt.show()
plt.savefig('application_clusters.png')

print('Clusters -- share per class category')
print('cluster \t| interview \t| hired \t| pre-interview')
for l in range(db.labels_.max()):
    cond = np.bincount(y[np.where(db.labels_ == l)])
    print(l, cond / cond.sum())

dd = {}
for factor in demographic_data.columns:
    if factor == 'age':
        continue
    for e in demographic_data[factor].unique():
        if e is np.nan:
            cond = np.bincount(y[np.where(data.x_trn[factor].isna())])
            e = 'nan'
        else:
            cond = np.bincount(y[np.where(data.x_trn[factor] == e)])
        shares = cond / cond.sum()
        if len(shares) < 1:
            continue

        dd['%s_%s' % (factor, e)] = shares
df = pd.DataFrame.from_dict(dd, orient='index')
df.sort_values(by=2).loc[:,2]


