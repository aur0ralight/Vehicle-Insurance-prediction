'''
 CREATING MODEL AND SAVING IN PICKLE FORMAT
'''
# import pickle
# import json
# import pandas as pd
# from sklearn.svm import SVC
#
# from utils.dataloader import DataLoader
# from settings.constants import TRAIN_CSV
#
# with open('settings/specifications.json') as f:
#     specifications = json.load(f)
#
# raw_train = pd.read_csv(TRAIN_CSV)
# x_columns = specifications['description']['X']
# y_column = specifications['description']['y']
#
# x_raw = raw_train[x_columns]
#
# loader = DataLoader()
# loader.fit(x_raw)
# X = loader.load_data()
# y = raw_train.Response
#
# model = SVC()
# model.fit(X, y)
# with open('models/SVC.pickle', 'wb')as f:
#     pickle.dump(model, f)

'''
 CHECKING PICKLE MODEL ON VALIDATION DATA
'''
# import pickle
# import json
# import pandas as pd
# from sklearn.svm import SVC
#
# from utils.dataloader import DataLoader
# from settings. constants import VAL_CSV
#
#
# with open('settings/specifications.json') as f:
#     specifications = json.load(f)
#
# x_columns = specifications['description']['X']
# y_column = specifications['description']['y']
#
# raw_val = pd.read_csv(VAL_CSV)
# x_raw = raw_val[x_columns]
#
# loader = DataLoader()
# loader.fit(x_raw)
# X = loader.load_data()
# y = raw_val.Response
#
# loaded_model = pickle.load(open('models/SVC.pickle', 'rb'))
# print(loaded_model.score(X, y))


'''
 WORKING WITH DATA BEFORE WRITING FEATURES IN specifications.json
'''
# import numpy as np
# import pandas as pd
# import re as re
#
# from settings.constants import TRAIN_CSV, VAL_CSV
# from sklearn.preprocessing import LabelEncoder
# pd.set_option("display.max_rows", None, "display.max_columns", None)
#
# train = pd.read_csv(TRAIN_CSV, header=0)
# val = pd.read_csv(VAL_CSV, header=0)
# full_data = [train, val]
#
# print(train.info())
#
# # normalization
# # max_value = train['Annual_Premium'].max()
# # min_value = train['Annual_Premium'].min()
# # train['Annual_Premium'] = (train['Annual_Premium'] - min_value) / (max_value - min_value)
# # train['Annual_Premium'] = (train['Annual_Premium']-train['Annual_Premium'].mean())/train['Annual_Premium'].std()
#
# # TREATMENT OF SKEWED COLUMNS
# # train["Age"] = np.sqrt(train["Age"])
# # train["Annual_Premium"] = np.sqrt(train["Annual_Premium"])
#
# # drop columns
# train = train.drop(['Region_Code','Annual_Premium','id','Vintage'],axis=1)
#
# # replace values
# train['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
# train['Vehicle_Damage'].replace({'No': 0, 'Yes': 1}, inplace=True)
# train['Vehicle_Age'].replace({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}, inplace=True)
#
# # new features
# train['Insured_Damage'] = train['Vehicle_Damage'] - train['Previously_Insured'] + 1
#
# # binning
# train['Age'] = pd.qcut(train['Age'], 5)
# # print(train[['Response', 'Age_categorical']].groupby(['Age_categorical'], as_index=False).mean())
#
# train['Policy_Sales_Channel'] = pd.cut(train['Policy_Sales_Channel'], 6)
# # print(train[['Response', 'Policy_Sales_categorical']].groupby(['Policy_Sales_categorical'], as_index=False).mean())
#
# # 0.117463 -0.134522
# # encode labels
# le = LabelEncoder()
#
# le.fit(train['Age'])
# train['Age'] = le.transform(train['Age'])
# le.fit(train['Policy_Sales_Channel'])
# train['Policy_Sales_Channel'] = le.transform(train['Policy_Sales_Channel'])
#
# # print(train.corr())
# # print(train.info())



'''
 IMPORTING DATA
'''
# import numpy as np
# import pandas as pd
#
# train_set = pd.read_csv('trainfull.csv')
# df_shuffled=train_set.iloc[np.random.permutation(train_set.index)].reset_index(drop=True)
# result = df_shuffled.assign(id = np.arange(1, df_shuffled.shape[0]+1))
# df_tr = result[:19000]
# df_val = result[19000:28000].assign(id = np.arange(1, 9001))
#
# a = df_tr.to_csv('train.csv', index=False)
# b = df_val.to_csv('val.csv', index=False)