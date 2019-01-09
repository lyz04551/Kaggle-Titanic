import numpy as np
import re
import pandas
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.ensemble import GradientBoostingClassifier


def get_title(name):
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ''

if __name__=='__main__':
    titanic=pandas.read_csv('data/train.csv')
    titanic_test=pandas.read_csv('data/test.csv')
    # print(titanic.head())   输出前几行的信息,具体表格里的内容
    # print(titanic.describe())  输出数据的count，mean，std，min各种数据信息
    titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())    #填充数据  还有其他方式么？？？
    titanic.loc[titanic['Sex']=='male','Sex']=0
    titanic.loc[titanic['Sex']=='female','Sex']=1

    titanic['Embarked']=titanic['Embarked'].fillna('S')
    titanic.loc[titanic['Embarked']=='S','Embarked']=0
    titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
    titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

    titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].mean())  # 填充数据  还有其他方式么？？？
    titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
    titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1

    titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
    titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
    titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
    titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2

    titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())
    # print(titanic_test.describe())
    # titanic['Died']=1-titanic['Survived']

    predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    # alg = LinearRegression()

    # alg.fit(titanic[predictors],titanic['Survived'])    自己写的
    # prediction=alg.predict(titanic_test[predictors])
    # print(prediction)

    #交叉验证 kfold
    # kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
    # predictions=[]
    # for train,test in kf:
    #     train_predictors=(titanic[predictors].iloc[train,:])
    #     # print(train_predictors)
    #     train_target=titanic['Survived'].iloc[train]
    #     alg.fit(train_predictors,train_target)
    #     test_predictions=alg.predict(titanic[predictors].iloc[test,:])
    #     predictions.append(test_predictions)
    #     # print(predictions)
    # predictions=np.concatenate(predictions,axis=0)
    # predictions[predictions>.5]=1
    # predictions[predictions<=.5]=0
    # # print(predictions,predictions.shape)
    # accuary=sum(predictions[predictions==titanic['Survived']])/len(predictions)
    # print(accuary)

    #线性回归交叉验证分数
    # alg=LinearRegression()
    # score=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=3)
    # print(score)

    #逻辑回归交叉验证分数
    # alg=LogisticRegression(random_state=1)
    # score=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=5)
    # print(score.mean())
    # alg.fit(titanic[predictors],titanic['Survived'])
    # prediction=alg.predict(titanic_test[predictors])
    # print(prediction)

    #随机森林交叉验证
    # alg=RandomForestClassifier(random_state=1,n_estimators=30,min_samples_split=8,min_samples_leaf=4)
    # kf = cross_validation.KFold(titanic.shape[0], n_folds=5, random_state=1)
    # scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
    # alg.fit(titanic[predictors],titanic['Survived'])
    # prediction=alg.predict(titanic_test[predictors])
    # print(scores.mean())
    # result = pandas.DataFrame(
    #      {'PassengerId': titanic_test['PassengerId'].as_matrix(), 'Survived': prediction.astype(np.int32)})
    # result.to_csv("RandomForestClassifier_predictions2.csv", index=False)

    #特征工程 处理数据
    titanic['FamilySize']=titanic['SibSp']+titanic['Parch']
    titanic['NameLength']=titanic['Name'].apply(lambda x:len(x))

    title=titanic['Name'].apply(get_title)
    # print(pandas.value_counts(title))
    titles_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Major': 7,
                     'Col': 7, 'Mlle': 8, 'Mme': 8, 'Don': 9, 'Lady': 10, 'Countess': 10, 'Jonkheer': 10, 'Sir': 9,
                     'Capt': 7, 'Ms': 2}
    for k,v in titles_mapping.items():
        title[title==k]=v
    titanic['Title']=title

    # predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'NameLength']
    # selector=SelectKBest(f_classif,k=5)
    # selector.fit(titanic[predictors], titanic['Survived'])
    # scores = -np.log10(selector.pvalues_)
    # plt.bar(range(len(predictors)), scores)
    # plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    # plt.show()

    # predictors = ['Pclass', 'Sex', 'Age', 'Fare']
    # alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=8,min_samples_leaf=4)
    # kf = cross_validation.KFold(titanic.shape[0], n_folds=5, random_state=1)
    # scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
    # print(scores.mean())

    algorithms=[
        [GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),['Pclass', 'Sex', 'Age', 'Fare']],
        [LogisticRegression(random_state=1),['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    ]
    apredictions=[]
    for alg,predictors in algorithms:
        alg.fit(titanic[predictors],titanic['Survived'])
        test_predictions=alg.predict_proba(titanic[predictors].astype(float))[:,1]
        apredictions.append(test_predictions)

    predictions=(apredictions[0]+apredictions[1])/2
    # print(predictions)
    # predictions[predictions>.5]=1
    # predictions[predictions<=.5]=0
    # accuary=sum(predictions[predictions==titanic['Survived']])/len(predictions)
    # print(accuary)