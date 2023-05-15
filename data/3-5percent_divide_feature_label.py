import pandas as pd

""" df = pd.read_csv('seed=12=01-12-5-percentclean.csv')
label = df.loc[:, [' Label']] # 只取得 label column
feature = df.drop(columns=[' Label']) # 除了 label column 以外
label.to_csv('seed=12=01-12-5-percentclean-label.csv', index = False)
feature.to_csv('seed=12=01-12-5-percentclean-feature.csv', index = False)
 """
df = pd.read_csv('seed=12=03-11-5-percentclean.csv')
label = df.loc[:, [' Label']] # 只取得 label column
feature = df.drop(columns=[' Label']) # 除了 label column 以外
label.to_csv('seed=12=03-11-5-percentclean-label.csv', index = False)
feature.to_csv('seed=12=03-11-5-percentclean-feature.csv', index = False)