## Karar Ağacı Kullanarak Bireylerin Cinsiyetlerini Tahminleyen Program 


#Kullanacağımız kütüphanelerimizi yüklüyoruz
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

## 1.Veri Toplama

#Excel Üzerindeki Veri Setimizi Çalışma Sayfamıza Aktarıyoruz
df=pd.read_csv("Bar-Genders.csv",delimiter=";")

## 2.Veri Analizi

df

df.info()

## 3.Sonuçların Görselleştirilmesi

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kfold=KFold(10,shuffle=True,random_state=True)

#kategorik verileri (metin etiketleri) sayısal temsillere dönüştürme
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
        
X = df.drop("Gender", axis=1)
y = df.Gender

X=df.drop("Gender",axis=1)
y=df.Gender
k=0
accuracy=[]
for train,test in kfold.split(df):
    print(k,len(train),len(test))
    k+=1
    trainSet=df.iloc[train]
    testSet=df.iloc[test]

    dt=tree.DecisionTreeClassifier(criterion="gini")
    dt.fit(trainSet.drop("Gender",axis=1),trainSet.Gender)

    pred=dt.predict(testSet.drop("Gender",axis=1))
    score=accuracy_score(testSet.Gender,pred)
    accuracy.append(score)
    print("Accuracy:"+str(score))


## 4.Sonuçların Yorumlanması 

print(sum(accuracy)/len(accuracy))
from matplotlib import pyplot as plt
tree.plot_tree(dt)
plt.show()

