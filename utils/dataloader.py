import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):

        # replace value
        # self.dataset['Gender'] = self.dataset['Gender'].replace('Male', 0)
        # self.dataset['Gender'] = self.dataset['Gender'].replace('Female', 1)
        # self.dataset['Vehicle_Damage'] = self.dataset['Vehicle_Damage'].replace('No', 0)
        # self.dataset['Vehicle_Damage'] = self.dataset['Vehicle_Damage'].replace('Yes', 1)
        # self.dataset['Vehicle_Age'] = self.dataset['Vehicle_Age'].replace('< 1 Year', 0)
        # self.dataset['Vehicle_Age'] = self.dataset['Vehicle_Age'].replace('1-2 Year', 1)
        # self.dataset['Vehicle_Age'] = self.dataset['Vehicle_Age'].replace('> 2 Years', 2)

        # columns combination
        # self.dataset['Insured_Damage'] = self.dataset['Vehicle_Damage'] - self.dataset['Previously_Insured'] + 1

        # binning with cut
        # self.dataset['Age'] = pd.qcut(self.dataset['Age'], 5)
        # self.dataset['Policy_Sales_Channel'] = pd.cut(self.dataset['Policy_Sales_Channel'], 6)

        # drop columns
        drop_elements = ['Region_Code', 'id', 'Vintage']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        # encode labels
        le = LabelEncoder()

        # le.fit(self.dataset['Age'])
        # self.dataset['Age'] = le.transform(self.dataset['Age'])
        # le.fit(self.dataset['Policy_Sales_Channel'])
        # self.dataset['Policy_Sales_Channel'] = le.transform(self.dataset['Policy_Sales_Channel'])
        le.fit(self.dataset['Vehicle_Age'])
        self.dataset['Vehicle_Age'] = le.transform(self.dataset['Vehicle_Age'])
        le.fit(self.dataset['Vehicle_Damage'])
        self.dataset['Vehicle_Damage'] = le.transform(self.dataset['Vehicle_Damage'])
        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        return self.dataset