str = "Hello World!" 
print(str[::-1])
###############
str = input()
print(2*int(str))
##############
str = input()
print(str[str.find(' ') + 1:]+ ' ' + str[:str.find(' ')] )
####################################
str = input()
print(str[:str.find('@')])
###################################
k = 0
str = input()
for i in range(len(str)):
    if str[i] == '"':
        k += 1
    if k == 1 or str[i] == '"':
        print(str[i], end = '')
######################################
str = '+7 (812) 134-12-324'
for x in str:
  if x not in "+1234567890":
    str = str.replace(x,'')
print(str)
######################################
str = 'а роза упала на лапу Азора'
num_test = str[::-1]
print(str.lower().replace(' ', ''))
print(num_test.lower().replace(' ', ''))
if  str.lower().replace(' ', '') == num_test.lower().replace(' ', ''):
    print("Is a palindrome")
else:
    print("Is NOT a palindrome")
#################################
for i in range(1, 10):
    for j in range(10):
        for k in range(10):
            number = i * 100 + j * 10 + k
            if i**3 + j**3 + k**3 == number:
                print(number)
#################################
import math
import string
n = int(input())
for i in range(1, n + 1):
  a = str(i)
  b = str(i*i)
  if b[-len(a):] == a:
     print(b,' ', a)
####################################
count = 0
for i in range(13):
  for j in range(11):
    for k in range(9):
      if 185 == i*15+j*17+k*21:
        count += 1
        print(i,j,k)
print('Всего способов - ', count)
####################################
#надо файлик загрузить/ про библиотеку
import numpy as np
A = np.genfromtxt('lib.txt', dtype = None, delimiter = ",")
#A = np.array(['Мцыри', 'Преступление и наказание', 'Кортик', 'Бородино', 'Молодая гвардия', 'Евгений Онегин', 'Герой нашего времени', 'Идиот', 'Зойкина квартира', 'Демон', 'Собачье Сердце', 'Сборник романов Достоевского', 'Стихотворения Лермонтова', 'Белая гвардия', 'Мастер и Маргарита', 'Лучшие произведения М.Ю. Лермонтова', 'Горячий снег', 'Дикая собака Динго', 'Беглец', 'Стихотворения Пушкина', 'Василий Теркин', 'Выхожу один я на дорогу', 'Война и Мир', 'Анна Каренина', 'Кинжал', 'Севастопольские рассказы', 'Бал', 'Ангел', 'Романы Толстого', 'Князь Серебряный','Сердца Трех','По Ком Звонит Колокол', 'Марк Твен', 'Малыш и Карлсон', 'Великий Гетсби', 'Маугли', 'Мио Мой Мио', 'На западном фронте без перемен', 'Портрет Дориана Грея', 'Мартин Иден', 'Собор Парижской Богоматери', 'Отверженные', 'Белый Клык', 'Человек, который смеется', 'Детективы Агаты Кристи', 'Фауст', 'Лучшие стихотворения Киплинга', 'Рубаи Омара Хайяма', 'Сонеты Уильяма Шекспира', 'Гомер: Иллиада', 'Гомер: Одиссея', 'Ворон', 'Уильям Блейк', 'Гораций', 'Басё', 'Сборник стихотворений Гёте', 'Шиллер. Избраннные произведения', 'Сайгё', 'Сборник Роберта Бернса', 'Омар Хайям'])
n = A.size
k = 0
for i in range(int(n/2 - 1), int(3/4 * n)):
  if i % 3 == 0:
    if k <= 5:
      print(A[i])
##################################
#температура
import numpy as np
!head stockholm_tmp.dat
data = np.genfromtxt('stockholm_tmp.dat')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = [12,12])
y = np.array([1.,2.,1.,2.,1.,2.,1.,2.,1.,2.,1.,2.])
for a in range(12):
  mask_mon = data[:, 1] == a+1
  y[a] = (np.mean(data[mask_mon,3]))
ax.bar(np.linspace(1,12,12),y)
ax.axis('tight')
ax.set_xlabel('Mounth')
ax.set_ylabel('Temperature')
print(y)
####################################
#ТИТАНИК
from pandas.core.common import maybe_iterable_to_list
import numpy as np
import pandas as pd
data = pd.read_csv('titanic.csv', delimiter=',')
# print(data.head())
#data.notnull()
#data.isna()
# data["Age"].head()
# data[data["Survived"] == 0].head()
# data.fillna("ПРОПУСК").head()
# data["Relatives"] = data["SibSp"] + data["Parch"]
data[["Sex", "Survived"]].pivot_table(index=["Sex"], columns=["Survived"], aggfunc=len)
data[data["Survived"]==1][["Sex", "Age"]].pivot_table(values=["Age"], index=["Sex"], aggfunc="mean")
data.groupby("Pclass").mean()["Age"]
# data.describe()
# data.groupby("Pclass").describe()["Age"]
####################################
import numpy as np
import pandas as pd
normal_sample= np.random.normal(loc=0.0, scale = 1.0, size=10000)
random_sample= np.random.random(size = 10000)
gamma_sample= np.random.gamma(2, size = 10000)
df = pd.DataFrame({'normal' :normal_sample,
                'random': random_sample,
                    'gamma': gamma_sample})
###########################################
df.describe()
############################################
import matplotlib .pyplot as plt
import seaborn as sns

plt.figure()
_ = plt.boxplot(df['normal'],whis=[0,100])
##########################################
data["Age"].mean()
####################################
data[["Sex", "Survived", "Age"]].pivot_table(values=["Age"], index=["Sex"], columns=["Survived"], aggfunc="mean")
###############################################
data.hist();
###########################################
data["Age"].plot("kde", xlim=(data["Age"].min(), data["Age"].max()));
################################
data["Sex"].value_counts().plot(kind="pie", figsize=(7, 7), fontsize=20);
##############################
data["Pclass"].value_counts().plot(kind="pie", figsize=(7, 7), fontsize=20);
#############################
data[["Sex", "Survived"]].pivot_table(index=["Sex"], columns=["Survived"], aggfunc=len).plot(kind="bar");



#АНИМЕ
##########################################
import numpy as np
import pandas as pd
data = pd.read_csv('anime.csv', delimiter=',')
# data = data.columns.str.replace('?',' ',regex=True)
data
##############################
data = pd.read_csv("anime.csv", sep=",",na_values=["?"])
na_values=["?"]
data
###########
data.head(10)
############################
data.dtypes
############################
data.info()
#############################
data.columns = data.columns.str.lower()
data
##############################
data['voters']=data['voters'].str.replace(',','')
data['voters'] = data['voters'].astype('int64')
perc =[.90]
data.describe(percentiles = perc, include = ['float64','int64'])
#####################################
data.groupby(['production', 'title']).sum()
#####################################
data.fillna({'episodes':0,'source':' ',	'genre': ' ',	'airdate':' ',	'rating' :' ',	'voters': 0,	'theme': ' '})
#####################################



