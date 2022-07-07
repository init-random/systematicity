import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.murmurhash import murmurhash3_32
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import math

class ApplicationData(object):
    def __init__(self, fn):
        self.data_fn = fn
        self._data_init()
        self._raw_splits()
        self._demographic_features()
        self._x_features()

    def _data_init(self):
        self.data = pd.read_csv(self.data_fn, dialect='unix', delimiter=',')
        self.n_rows = self.data.count()
        self.y, _ = self.data['application_status'].factorize()
        self.y_map = _.tolist()
        # rm y from data
        del self.data['application_status']
        # rm id fields
        rm_id_fields = []
        for k, cols in pd.concat((self.data.dtypes, self.data.count()), axis=1).iterrows():
            if len(self.data[k].unique()) == cols[1]:
                print(k, cols[1])
                rm_id_fields.append(k)
        for k in rm_id_fields:
            del self.data[k]

    def _raw_splits(self):
        slice1 = np.where(self.data['candidate_demographic_variable_3'] == 0)
        slice2 = np.where(self.data['candidate_demographic_variable_3'] != 0)
        a_trn, a_tst, ay_trn, ay_tst = train_test_split(self.data.loc[slice1], self.y[slice1], shuffle=True, test_size=0.1)
        b_trn, b_tst, by_trn, by_tst = train_test_split(self.data.loc[slice2], self.y[slice2], shuffle=True, test_size=0.1)
        self.x_trn_raw, self.x_tst_raw = pd.concat((a_trn, b_trn), axis=0), pd.concat((a_tst, b_tst), axis=0)
        self.x_trn_raw.reset_index(drop=True, inplace=True)
        self.x_tst_raw.reset_index(drop=True, inplace=True)
        self.y_trn, self.y_tst = np.hstack((ay_trn, by_trn)), np.hstack((ay_tst, by_tst))

    def _demographic_features(self):
        train, test = self.x_trn_raw, self.x_tst_raw
        # demographic information
        self.demographic_cols = [c for i, c in enumerate(self.data.columns) if
                            'demographic' in c or c in ['age', 'ethnicity', 'gender']]
        float_cols = []
        obj_cols = []
        for k, dt in train.loc[:, self.demographic_cols].dtypes.items():
            if dt == np.float64 or dt == np.int64:
                float_cols.append(k)
            else:
                obj_cols.append(k)
        demographic_data = []
        demographic_data_tst = []
        for c in float_cols:
            m = train[c].median()
            if m == 0.5 and train[c].max == 1.0:
                m = round(m)
            demographic_data.append(train[c].fillna(m))
            demographic_data_tst.append(test[c].fillna(m))

        for c in obj_cols:
            demographic_data.append(pd.get_dummies(train[c], prefix=c))
            demographic_data_tst.append(pd.get_dummies(test[c], prefix=c))
        self.demographic_data = pd.concat(demographic_data, axis=1)
        self.demographic_data_tst = pd.concat(demographic_data_tst, axis=1)
        self.nn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
        self.demographic_data.reset_index(drop=True, inplace=True)
        self.nn.fit(self.demographic_data)
        neighbors = self.nn.kneighbors(self.demographic_data, return_distance=False)[:, 1:]
        for i, r in enumerate(train.isna().iterrows()):
            n = neighbors[i, :]
            med = train.loc[n, float_cols].median()
            for k, v in r[1].iteritems():
                if k not in float_cols:
                    continue
                if v and train.loc[:, k].dtype != np.object0:
                    if not pd.isna(med[k]):
                        m = med[k]
                        if m == 0.5:
                            m = round(m)
                        self.demographic_data.loc[r[0], k] = m
        # update test data based on training values
        neighbors = self.nn.kneighbors(self.demographic_data_tst, return_distance=False)[:, :-1]
        for i, r in enumerate(test.isna().iterrows()):
            n = neighbors[i, :]
            med = train.loc[n, float_cols].median()
            for k, v in r[1].iteritems():
                if k not in float_cols:
                    continue
                if v and test.loc[:, k].dtype != np.object0:
                    if not pd.isna(med[k]):
                        m = med[k]
                        if m == 0.5:
                            m = round(m)
                        self.demographic_data_tst.loc[r[0], k] = m

    def _x_features(self):
        train, test = self.x_trn_raw, self.x_tst_raw
        z_cols = [c for c in self.data.columns if c not in self.demographic_cols]
        float_cols = []
        for k, dt in train.loc[:, z_cols].dtypes.items():
            if dt == np.float64 or dt == np.int64:
                float_cols.append(k)

        z_data = []
        z_data_tst = []
        for c in float_cols:
            m = train[c].median()
            if m == 0.5 and train[c].max() == 1.0:
                m = round(m)
            z_data.append(train[c].fillna(m))
            z_data_tst.append(test[c].fillna(m))

        z_data = pd.concat(z_data, axis=1)
        z_data_tst = pd.concat(z_data_tst, axis=1)
        self.nn.fit(z_data)
        neighbors = self.nn.kneighbors(z_data, return_distance=False)[:, 1:]

        for i, r in enumerate(train.isna().iterrows()):
            n = neighbors[i, :]
            med = train.loc[n, float_cols].median()
            for k, v in r[1].iteritems():
                if k not in float_cols:
                    continue
                if v and train.loc[:, k].dtype != np.object0:
                    if not pd.isna(med[k]):
                        m = med[k]
                        if m == 0.5:
                            m = round(m)
                        z_data.loc[r[0], k] = m
        # z_data_tst.reset_index(inplace=True, drop=True)
        # self.nn.fit(z_data)
        neighbors = self.nn.kneighbors(z_data_tst, return_distance=False)[:, :-1]
        for i, r in enumerate(test.isna().iterrows()):
            n = neighbors[i, :]
            med = train.loc[n, float_cols].median()
            for k, v in r[1].iteritems():
                if k not in float_cols:
                    continue
                if v and test.loc[:, k].dtype != np.object0:
                    if not pd.isna(med[k]):
                        m = med[k]
                        if m == 0.5:
                            m = round(m)
                        z_data_tst.loc[r[0], k] = m

        self.x_trn = pd.concat((z_data, self.demographic_data), axis=1)
        self.x_tst = pd.concat((z_data_tst, self.demographic_data_tst), axis=1)


class ApplicationClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_classifiers=25):
        self.n_classifiers = n_classifiers

    def fit(self, X, y):
        self.rf_clf = [RandomForestClassifier(n_estimators=np.random.randint(10, 50, 1)[0],
                                         max_depth=np.random.randint(3, 20, 1)[0],
                                         criterion=['gini', 'entropy'][np.random.randint(0, 2, 1)[0]])
                  for _ in range(self.n_classifiers)]
        for _ in range(self.n_classifiers):
            self.rf_clf[_].fit(X, y)
        f1_global = []
        for clf in self.rf_clf:
            yhat = clf.predict(X)
            f1_global.append(f1_score(y, yhat, average='macro'))
        self.best_clf = self.rf_clf[np.argmax(f1_global)]

    def predict(self, x, mode='random_clf'):
        if mode not in [None, 'random_clf', 'jitter_prob']:
            raise 'Unknown mode.'
        if mode == 'random_clf':
            ys = []
            for _x in x.iterrows():
                h = murmurhash3_32(str(_x)) % 2**31
                np.random.seed(h)
                clf = self.rf_clf[np.random.randint(0, self.n_classifiers, 1)[0]]
                ys.append(clf.predict(_x[1].to_frame().T)[0])
            yhat = np.array(ys)
            return yhat
        if mode is None:
            clf = self.best_clf
            yhat = clf.predict(x)
            return yhat
        else:
            h = murmurhash3_32(str(x)) % 2**31
            np.random.seed(h)
            clf = self.best_clf
            yhat = clf.predict_proba(x)
            jitter = np.random.uniform(1.0e-6, 0.1, yhat.shape)
            yhat *= jitter
            return yhat.argmax(axis=1)


def eval(model, x_tst, y_tst):
    idx_a = np.where(x_tst['candidate_demographic_variable_3'] == 0)
    idx_b = np.where(x_tst['candidate_demographic_variable_3'] != 0)

    yhat_none = model.predict(x_tst, mode=None)
    yhat_rand = model.predict(x_tst, mode='random_clf')
    yhat_prob = model.predict(x_tst, mode='jitter_prob')

    print('Systematicity treatment:')
    print('None:')
    print('interview \t| hired \t| pre-interview')
    print('Demo var3=0', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst[idx_a], yhat_none[idx_a], average=None)]))
    print('Demo var3!=0', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst[idx_b], yhat_none[idx_b], average=None)]))
    print('Global', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst, yhat_none, average=None)]))

    print('Random classifier:')
    print('interview \t| hired \t| pre-interview')
    print('Demo var3=0', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst[idx_a], yhat_rand[idx_a], average=None)]))
    print('Demo var3!=0', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst[idx_b], yhat_rand[idx_b], average=None)]))
    print('Global', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst, yhat_rand, average=None)]))

    print('Jitter probability:')
    print('interview \t| hired \t| pre-interview')
    print('Demo var3=0', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst[idx_a], yhat_prob[idx_a], average=None)]))
    print('Demo var3!=0', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst[idx_b], yhat_prob[idx_b], average=None)]))
    print('Global', ' \t|'.join(['%.4f' % x for x in f1_score(y_tst, yhat_prob, average=None)]))


def main():
    data_fn = 'data/data.csv'
    data = ApplicationData(data_fn)
    model = ApplicationClassifier()
    model.fit(data.x_trn, data.y_trn)
    eval(model, data.x_tst, data.y_tst)


if __name__ == '__main__':
    main()

