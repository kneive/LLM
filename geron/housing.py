from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url="https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data();


# feeling out the data
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(12,8))
plt.show()

# test set
import numpy as np
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) <  test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

# create an income category

housing["income_cat"] = pd.cut( housing["median_income"],
                                bins=[0.,1.5,3.0,4.5,6., np.inf],
                                labels=[1,2,3,4,5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()


# create splitter

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits=[]
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# acces first split
strat_train_set, strat_test_set = strat_splits[0]

# alternative
from sklearn.model_selection import train_test_split
alt_strat_train_set, alt_strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# test
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
alt_strat_test_set["income_cat"].value_counts() / len(alt_strat_test_set)

# drop "income_cat"
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
for set_ in (strat_train_set, strat_test_set, alt_strat_train_set, alt_strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
