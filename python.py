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
import matplotlib as plt
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import axes3d

genresList = list(data['genre'].str.split(','))
genres = dict()
listGeneres = sum(genresList, [])
fig, ax = mpl.subplots()
fig.set_figwidth(22)   
fig.set_figheight(6) 
ax.bar(list(sorted(set(listGeneres))), [listGeneres.count(x) for x in sorted(set(listGeneres))], 1, 50)

fig2, bx = mpl.subplots()
fig2.set_figwidth(40)   
fig2.set_figheight(10)

company = dict((x, list(data['production']).count(x)) for x in set(data['production']))
sorted_company = sorted(company.items(), key=lambda x: x[1])
print(dict(sorted_company))
bx.bar(dict(sorted_company).keys(), dict(sorted_company).values())
mpl.xticks(rotation=90) 

fig3, cx = mpl.subplots()
fig3.set_figwidth(40)   
fig3.set_figheight(10)


ep = dict((x, list(data['episodes'].dropna()).count(x)) for x in set(data['episodes'].dropna()))
ep_sorted = sorted(ep.items(), key=lambda x: x[0])
print(ep_sorted)
cx.bar(dict(ep_sorted).keys(), dict(ep_sorted).values())




############################################
import numpy as np
import pandas as pd
normal_sample = np.random.normal(loc = 0.0, scale = 1.0, size = 10000)
random_sample = np.random.random(size = 10000)
gamma_sample = np.random.gamma(2, size = 10000)
df = pd.DataFrame({'normal': normal_sample, 'random': random_sample, 'gamma': gamma_sample})
df.head()
df.describe()
#######
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
_ = plt.boxplot([df['normal'], df['random'], df['gamma']], whis = [10, 90])
####################
dff = pd.DataFrame(data = df, columns = ["normal", "random", "gamma"])
sns.boxplot(x = None, y = None, data = dff)
bp = sns.boxplot(x = "variable", y = 'value', data = pd.melt(dff), whis = [0, 100])

#########################

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
ax = axes3d.Axes3D(plt.figure())
i = np.arange(-1,1,0.01)
X,Y=np.meshgrid(i,i)
Z = X**2 + Y**3
ax.plot_wireframe(X,Y,Z,rstride=3,cstride=3)
plt.show()

#############################
from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pit
fig = pit.figure()
ax = fig.add_subplot(111, projection = '3d')

####################
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
x = y = np.linspace(-3,3,74)
X,Y=np.meshgrid(x,y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(4*R)/R
fig, ax = plt.subplots(1,3,figsize=(14,4), subplot_kw=dict(projection = '3d'))

norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())
p=ax[0].plot_surface(X,Y,Z,rstride=1,cstride=1,linewidth=0,
antialiased = True, norm = norm, cmap=mpl.cm.Blues)
cb = fig.colorbar(p, ax=ax[0], shrink=0.6)
ax[0].set_xlabel("$x$", fontsize = 16)
ax[0].set_ylabel("$y$", fontsize = 16)
ax[0].set_zlabel("$z$", fontsize = 16)

ax[1].plot_wireframe(X,Y,Z,rstride=3,cstride=3, color = "yellow")
ax[1].set_title("plot_wireframe")

ax[2].contour(X,Y,Z,zdir="z",offset=0, norm=norm,cmap=mpl.cm.jet)
ax[2].contour(X,Y,Z,zdir="y",offset=3, norm=norm,cmap=mpl.cm.Blues)
plt.show()

#######################################
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import axes3d


ax[1].plot_wireframe(X,Y,Z, rstride = 3, cstride = 3, color = "darkgrey")
ax[1].set_title("plot_wireframe")

import seaborn as sns
np.random.seed(1234)

v1 = pd.Series(np.random.normal(0,10,5000), name = 'v1')
v2 = pd.Series(2*v1+np.random.normal(60,15,5000),name = 'v2')

figure = mpl.figure()
mpl.hist(v1,alpha = 0.7, bins = np.arange(-50,150,5), label = 'v1')
mpl.hist(v2,alpha = 0.7, bins = np.arange(-50,150,5), label = 'v1')

from locale import normalize
mpl.figure()
mpl.hist([v1,v2], histtype="barstacked", density = True)
v3 = np.concatenate((v1, v2))
sns.kdeplot(v3)

mpl.figure()
sns.distplot(v3,hist_kws = {'color':'Teal'}, kde_kws ={'color':'Navy'})

mpl.figure()
sns.jointplot(v1,v2,alpha = 0.5, kind = "hex")


###############
from math import *

def temperature(t: float) -> float:
  return (atan(-0.0012*t**3+0.4*t**2+0.616*t+6120)+0.65*sin(0.24*t+1.23)-0.27*cos(0.21*t-0.17)-(sin(0.34*t+0.16))/(1+0.03*(t-370.5)**2))

x = range(0, 1000)
y = [temperature(t) for t in x]


plt.plot(x,y)
plt.show()

###########
from collections import Counter
import math
with open('orwell_1984.txt', encoding = 'utf8') as text:
  words_lenghts_counter = Counter(len(word) for word in text.read().split())

words_lenghts_counter

import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim([0,30])
ax.set_ylim([0,40000])

ax.set_xlabel("x")
ax.set_ylabel("y")

longestWordLength = max(words_lenghts_counter)

histogram = [words_lenghts_counter[wordLen] for wordLen in range(1, longestWordLength + 1)]

plt.hist(list(range(1, longestWordLength + 1)), weights = histogram, bins=longestWordLength+1, orientation='vertical')

######################

