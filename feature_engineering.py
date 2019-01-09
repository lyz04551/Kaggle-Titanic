import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation,metrics
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')



train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
all_data=pd.concat([train,test],ignore_index=True)
PassengerId=test['PassengerId']
# print(all_data.head())
#新增Title特征,从姓名中提取乘客的称呼归纳为6类
all_data['Title']=all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
# print(pd.value_counts(all_data['Title']))
Title_Dict={}
Title_Dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
all_data['Title']=all_data['Title'].map(Title_Dict)
# sns.barplot(x='Title',y='Survived',data=all_data,palette='Set3')
# plt.show()

#新增famaily特征,先计算FamilySize=Parch+SibSp+1，然后把FamilySize分成3类
#按生存率把FamilySize分为三类，构成FamilyLabel特征
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
def Fam_label(s):
    if (s>=2)&(s<=4):
        return 2
    elif ((s>4)&(s<=7))|(s==1):
        return 1
    elif (s>7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
# print(all_data['FamilyLabel'])
#新增Deck特征，先把Cabin空缺值填充为'Unknown'，再提取Cabin中的首字母构成乘客的甲板号。
all_data['Cabin']=all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
#新增TicketGroup特征，统计每个乘客的共票号数。
Ticket_Count=dict(all_data['Ticket'].value_counts())
all_data['TicketGroup']=all_data['Ticket'].apply(lambda x:Ticket_Count[x])
#按生存率把TicketGroup分为三类。
def Ticket_Label(s):
    if (s>=2) & (s<=4):
        return 2;
    elif ((s>4) & (s<=8) | (s==1)):
        return 1;
    elif (s>8):
        return 0
all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)

###缺失填充
#Age Feature：Age缺失量为263，缺失量较大，
# 用Sex, Title, Pclass三个特征构建随机森林模型，填充年龄缺失值。
age_df=all_data[['Age','Pclass','Sex','Title']]
age_df=pd.get_dummies(age_df)
known_age=age_df[age_df.Age.notnull()].as_matrix()
unknown_age=age_df[age_df.Age.isnull()].as_matrix()
y=known_age[:,0]
x=known_age[:,1:]
rfr=RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=-1)
rfr.fit(x,y)
predeictAges=rfr.predict(unknown_age[:,1::])
all_data.loc[all_data.Age.isnull(),'Age']=predeictAges

#Embarked Feature：Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，
# 因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C。
all_data['Embarked']=all_data['Embarked'].fillna('C')

#Fare Feature：Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3，
# 所以用Embarked为S，Pclass为3的乘客的Fare中位数填充。
fare = all_data[(all_data['Embarked']=='S')&(all_data['Pclass']==3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)

###同组识别
#因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。
# 把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组中存活率为1的设置为幸存组，推测处于遇难组
# 的女性和儿童幸存的可能性较低，处于幸存组的成年男性幸存的可能性较高
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())  #名字第一个单词
Surname_Count=dict(all_data['Surname'].value_counts())
all_data['FamilyGroup']=all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group = all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]

Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
# print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
# print(Survived_List)
#为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
train = all_data.loc[all_data['Survived'].notnull()]

test = all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)), 'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'

all_data=pd.concat([train,test])
all_data=all_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                   'Title', 'FamilyLabel', 'Deck', 'TicketGroup']]
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.as_matrix()[:, 1:]
y = train.as_matrix()[:, 0]

pipe = Pipeline([('select', SelectKBest(k=20)),
                 ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])
param_test = {'classify__n_estimators':list(range(20,50,2)),
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc')
gsearch.fit(X, y)
print(gsearch.best_params_, gsearch.best_score_)

select = SelectKBest(k=20)
clf = RandomForestClassifier(random_state=10, warm_start=True, n_estimators=26, max_depth=6, max_features='sqrt')
pipline = make_pipeline(select, clf)
pipline.fit(X, y)
cv_score = cross_validation.cross_val_score(pipline, X, y, cv=10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

predictions = pipline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived":predictions.astype(np.int32)})
submission.to_csv("mysubmission.csv", index=False)


