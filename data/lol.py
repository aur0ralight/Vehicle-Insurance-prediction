# import numpy as np
# import pandas as pd
#
# train_set = pd.read_csv('trainfull.csv')
# # df_shuffled=train_set.iloc[np.random.permutation(train_set.index)].reset_index(drop=True)
# # result = df_shuffled.assign(id = np.arange(1, df_shuffled.shape[0]+1))
# df_tr = train_set[:10000]
# df_val = train_set[10000:15000].assign(id = np.arange(1, 5001))
#
# a = df_tr.to_csv('train.csv', index=False)
# b = df_val.to_csv('val.csv', index=False)

# import numpy as np
# import pandas as pd
# import re as re
#
# train = pd.read_csv('train.csv')
# val = pd.read_csv('val.csv')
#
# drop_elements = ['Response', 'Region_Code', 'id', 'Vintage']
# train= train.drop(drop_elements, axis=1)
# print(train.groupby(train.columns.tolist(),as_index=False).size())
