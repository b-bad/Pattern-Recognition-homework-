# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
from naive_Bayes import NAIVE_BAYES
import pandas as pd
import numpy as np


# %%
train_data = pd.read_csv("../data/processed_train_data.csv")
train_data.head()


# %%
train_label = np.array(train_data["Survived"]).tolist()
train_data = np.array(train_data[['Sex', 'Age', 'Survived']]).tolist()
train_data


# %%
test_set = pd.read_csv("../data/processed_test_data.csv")
test_data = np.array(test_set[['Sex', 'Age']]).tolist()
test_data


# %%
NB = NAIVE_BAYES(train_data, train_label)


# %%
result = []

for i in test_data:
    result.append(NB.predict(i)[0])
# result


# %%
result = pd.DataFrame(result, columns=['Survived'])
output = pd.concat([test_set['PassengerId'], result], axis=1)


# %%
output.to_csv('../data/output2.csv', index=False)


# %%
print(result['Survived'].value_counts())


