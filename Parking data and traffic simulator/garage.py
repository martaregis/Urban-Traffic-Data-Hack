import pandas as pd
import numpy as np
from datetime import time
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid", palette='colorblind', context='paper')

DATAPATH = './All_data/DataDump_20160518/'
DATAFILE = 'zwolle_ddump_garage_parking_transactions_20160518_135145.csv'
USE_COLS = [0,1,3,4]
DATECOLS = [2,3]
SEP = ';'
TIMERESAMPLE = '30Min'

CPCTY = 'zwolle_ddump_garage_parking_garage_20160518_135205.csv'

parkdata = pd.read_csv(DATAPATH+DATAFILE, sep=SEP, usecols=USE_COLS, parse_dates=DATECOLS)
parkdata['park_time'] = (parkdata['end_parking_dt'] - parkdata['start_parking_dt'])/pd.Timedelta(1,'h')

capacity = pd.read_csv(DATAPATH+CPCTY, sep=SEP)
capacity = capacity.set_index('id')

parkdata ['count'] = 1
grouped = parkdata.groupby('garage_id')
foo = {}
for name, group in grouped:
    ts1 = group.loc[:, ['start_parking_dt', 'count']]
    ts1 = ts1.set_index('start_parking_dt')
    ts1 = ts1.resample(TIMERESAMPLE, how='sum')
    ts1 = ts1.sort_index()
    ts2 = group.loc[:, ['end_parking_dt', 'count']]
    ts2 = ts2.set_index('end_parking_dt')
    ts2 = ts2.resample(TIMERESAMPLE, how='sum')
    tss = ts2.align(ts1)
    ts2 = ts2.sort_index()
    bar = max(ts1.index[0], ts2.index[0])
    ts = ts1[bar:].fillna(0) - ts2[bar:].fillna(0)
    tsmin = ts.min()
    if tsmin.iloc[0] < 0:
        ts -= tsmin
    foo[name] = ts

df = pd.concat(foo)
df = df.reset_index()
df.columns = ['garage_id', 'time_interval', 'count']
df['time_interval'] = pd.DatetimeIndex(df['time_interval'])
df = df.pivot(index='time_interval', columns='garage_id', values='count')
df.columns = map(lambda x: capacity.loc[x, 'garage_nm'], df.columns)

df = df.sort_index()

slices = [time(i,j) for i in range(24) for j in range(0,60,15)]

means = pd.DataFrame(columns = df.columns)
stdvs = pd.DataFrame(columns = df.columns)

for i,j in zip(slices, slices[1:]+[slices[0]]):
    means = means.append(df.between_time(i,j).mean(), ignore_index=True)
    stdvs = stdvs.append(df.between_time(i,j).std(), ignore_index=True)

slices = map(lambda x: x.isoformat(), slices)
slices = pd.DatetimeIndex(slices)

means = means.set_index(slices)
stdvs = stdvs.set_index(slices)

upper = means + 1.96*stdvs
lower = means - 1.96*stdvs

f, ax = plt.subplots()
palette = sns.color_palette("husl", 8)
lgnd = []
for name, kolor in zip(means.columns, palette):
    lbl = name + ' (' + str(capacity.loc[capacity.garage_nm==name, 'capacity'].iloc[0]) + ')'
    plt.plot(range(len(means)), means[name], color=kolor, label=lbl)
    plt.fill_between(range(len(means)), lower[name], upper[name], color=kolor, alpha=.25)
    lgnd.append(ax.legend(loc='center left', bbox_to_anchor=(0.99, 0.5)))
xticks = [0,12,24,36,48,60,72,84]
times = [time(i,j) for i in range(24) for j in range(0,60,15)]
ax.set_xlim(0,95)
ax.set_xticks(xticks)
ax.set_xticklabels([times[i] for i in xticks])
ax.set_title('Average daily use of garages in Zwolle')
f.savefig('Plots/dailyavg.png', dpi=250, bbox_extra_artists=lgnd, bbox_inches='tight')

df.to_csv('./garage.csv')

# parkdata = parkdata[(parkdata.park_time > 1./12) & (parkdata.park_time < 18)]
