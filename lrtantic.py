import pandas
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def get_title(name):
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

if __name__=="__main__":

    titanic = pandas.read_csv('train.csv')
    # 年龄数据填充
    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
    # print(titanic.describe())

    # print(titanic['Sex'].unique()) 性别处理转为0 1
    titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
    titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
    # print(titanic['Sex'].unique())

    # print(titanic['Embarked'].unique())
    titanic['Embarked'] = titanic['Embarked'].fillna('S')
    titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
    titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
    titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
    # print(titanic['Embarked'].unique())

    predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

    #线性回归模型
    alg=LinearRegression()
    kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
    predictions=[]
    for train,test in kf:
        train_predictors=(titanic[predictors].iloc[train,:])
        # print(train_predictors)
        train_target=titanic['Survived'].iloc[train]
        alg.fit(train_predictors,train_target)
        test_predictions=alg.predict(titanic[predictors].iloc[test,:])
        predictions.append(test_predictions)
        # print(predictions)
    predictions=np.concatenate(predictions,axis=0)
    predictions[predictions>.5]=1
    predictions[predictions<=.5]=0
    accuary=sum(predictions[predictions==titanic['Survived']])/len(predictions)
    print(accuary)

    # 逻辑回归模型
    alg=LogisticRegression(random_state=1)
    scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=3)
    print(scores.mean())

    # 随机森林
    # alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
    # kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
    # scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
    # print(scores.mean())

    # 特征工程
    # titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
    # titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))
    #
    # titles = titanic['Name'].apply(get_title)
    # # print(pandas.value_counts(titles))
    # titles_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Major': 7,
    #                   'Col': 7, 'Mlle': 8, 'Mme': 8, 'Don': 9, 'Lady': 10, 'Countess': 10, 'Jonkheer': 10, 'Sir': 9,
    #                   'Capt': 7, 'Ms': 2}
    # for k, v in titles_mapping.items():
    #     titles[titles == k] = v
    # # print(pandas.value_counts(titles))
    # titanic['Title'] = titles
    # #选取有用的特征
    # predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'NameLength']
    # selector = SelectKBest(f_classif, k=5)
    # selector.fit(titanic[predictors], titanic['Survived'])
    # scores = -np.log10(selector.pvalues_)

    # plt.bar(range(len(predictors)), scores)
    # plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    # plt.show()

    # predictors=['Pclass','Sex','Fare','Title']
    # alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=8,min_samples_leaf=4)
    # kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
    # scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
    # alg.fit(titanic[predictors],titanic['Survived'])
    # # joblib.dump(alg, "train_model.pkl")
    # print(scores.mean())
    # print(alg)
    # print(kf)

    # titanic_test = pandas.read_csv('test.csv')
    # titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
    # titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
    # # print(titanic.describe())
    # del titanic_test['Cabin']
    # del titanic_test['Ticket']
    # # print(titanic['Sex'].unique()) 性别处理转为0 1
    # titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
    # titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
    # # print(titanic['Sex'].unique())
    #
    # # print(titanic['Embarked'].unique())
    # titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
    # titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
    # titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
    # titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
    #
    # titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']
    # titanic_test['NameLength'] = titanic_test['Name'].apply(lambda x: len(x))
    #
    # titles = titanic_test['Name'].apply(get_title)
    # # print(pandas.value_counts(titles))
    # titles_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Major': 7,
    #                   'Col': 7, 'Mlle': 8, 'Mme': 8, 'Don': 9, 'Lady': 10, 'Countess': 10, 'Jonkheer': 10, 'Sir': 9,
    #                   'Capt': 7, 'Ms': 2, 'Dona': 3}
    # for k, v in titles_mapping.items():
    #     titles[titles == k] = v
    # # print(pandas.value_counts(titles))
    # titanic_test['Title'] = titles
    # # print(titanic_test.info())
    # prediction=alg.predict(titanic_test[predictors])
    # print(prediction)
    # result = pandas.DataFrame(
    #     {'PassengerId': titanic_test['PassengerId'].as_matrix(), 'Survived': prediction.astype(np.int32)})
    # result.to_csv("RandomForestClassifier_predictions.csv", index=False)
