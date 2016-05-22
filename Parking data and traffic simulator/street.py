import pandas as pd
import numpy as np
import pdb

DATAPATH = './All_data/DataDump_20160518/'
DATAFILE = 'zwolle_ddump_street_parking_transactions_20160518_135210.csv'
USE_COLS = [0,1,3,4]
DATECOLS = [1]
SEP = ';'
TIMERESAMPLE = '30Min'

parkdata = pd.read_csv(DATAPATH+DATAFILE, sep=SEP, usecols=USE_COLS, parse_dates=DATECOLS)
parkdata['end_parking_dt'] = parkdata['start_parking_dt'] + pd.to_timedelta(parkdata['total_duration_sec'], unit='s')

#parkdata = parkdata[parkdata.meter_code < 20000]

parkdata ['count'] = 1
grouped = parkdata.groupby('meter_code')
foo = {}
for name, group in grouped:
    ts1 = group.loc[:, ['start_parking_dt', 'count']]
    ts1 = ts1.set_index('start_parking_dt')
    ts1 = ts1.resample(TIMERESAMPLE, how='sum')
    ts2 = group.loc[:, ['end_parking_dt', 'count']]
    ts2 = ts2.set_index('end_parking_dt')
    ts2 = ts2.resample(TIMERESAMPLE, how='sum')
    tss = ts2.align(ts1)
    ts = tss[1].fillna(0) - tss[0].fillna(0)
    tsmin = ts.min()
    if tsmin.iloc[0] < 0:
        ts -= tsmin
    foo[name] = ts

df = pd.concat(foo)
df = df.reset_index()
df.columns = ['meter_code', 'time_interval', 'count']
df = df.set_index('time_interval')
# df = df.set_index(['time_interval', 'meter_code'])
# df = df.sortlevel()

df = df.sort_index()


df.to_csv('./street.csv')


# parkdata = parkdata[(parkdata.park_time > 1./12) & (parkdata.park_time < 18)]
