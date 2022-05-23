import pandas as pd

interval = [i*1000 for i in range(43)] + [42003]
df_all = pd.DataFrame()
for i in range(len(interval) - 1):
    fileName = f'negative_random_comprehensive/negative_random_{interval[i]:05d}_{interval[i+1]:05d}.csv'
    df = pd.read_csv(fileName)
    df_all = pd.concat([df_all, df])
df_all.to_csv('negative_random_comprehensive.csv', index=False)
df_all