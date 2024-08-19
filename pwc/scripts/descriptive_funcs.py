import numpy as np

def sem(x):
    return np.std(x) / np.sqrt(len(x))


def meanRT(x):
    return (np.mean(x) - 1500) / 1000


def semRT(x):
    return sem(x) / 1000


def stdRT(x):
    sd = np.std(x) / 1000
    return sd if sd < 10 else np.nan


def exp_dur(x):
    return np.sum(x) / 1000 / 60


def pos_bias(x):
    return x.dropna().mean()


def pos_rt(x):
    pass


def count_left(x):
    return np.sum(x==0)


def count_right(x):
    return np.sum(x==1)

def count_timeouts(x):
    return x.isna().sum()



