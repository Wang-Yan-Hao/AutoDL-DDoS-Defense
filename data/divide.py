import pandas as pd

df = pd.read_csv('./output/01-12-five-percent-clean.csv')
label = df.loc[:, [' Label']] # Only get label name
feature = df.drop(columns=[' Label']) # Get other feature
label.to_csv('./output/01-12-five-percent-clean-feature.csv', index = False)
feature.to_csv('./output/01-12-five-percent-clean-feature.csv', index = False)

df = pd.read_csv('./output/03-11-five-percent-clean.csv')
label = df.loc[:, [' Label']]
feature = df.drop(columns=[' Label'])
label.to_csv('./output/03-11-five-percent-clean-feature.csv', index = False)
feature.to_csv('./output/03-11-five-percent-clean-feature.csv', index = False)