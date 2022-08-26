# Telco Churn Prediction

# İş Problemi

"""
Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

"""

# Veri Seti Hikayesi

"""
Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri 
sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, 
kaldığını veya hizmete kaydolduğunu gösterir.

"""
# Değişkenler
"""
21 Değişken 7043 Gözlem

CustomerId : Müşteri İd’si

Gender : Cinsiyet

SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)

Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama durumu.

Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)

tenure : Müşterinin şirkette kaldığı ay sayısı

PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)

MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)

InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)

OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)

StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir.

StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir.

Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)

PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)

PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))

MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar

TotalCharges : Müşteriden tahsil edilen toplam tutar

Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler

Her satır benzersiz bir müşteriyi temsil etmekte. Değişkenler müşteri hizmetleri, hesap ve demografik veriler hakkında bilgiler içerir.

Müşterilerin kaydolduğu hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler

Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı

"""
# Görev 1
# Keşifçi Veri Analizi

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz. Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# Adım 5: Aykırı gözlem var mı inceleyiniz.
# Adım 6: Eksik gözlem var mı inceleyiniz.

!pip3 install catboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/6_Makine_Ogrenmesi_YapayOgrenme/Müşteri_Terk_Tahmin_Modeli_Oluşturma/Telco-Customer-Churn.csv")
df.head()

"""
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport  \
0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No   
1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No   
2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No   
3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes   
4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No   
  
  
  StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn  
0          No              No  Month-to-month              Yes           Electronic check          29.850        29.85    No  
1          No              No        One year               No               Mailed check          56.950       1889.5    No  
2          No              No  Month-to-month              Yes               Mailed check          53.850       108.15   Yes  
3          No              No        One year               No  Bank transfer (automatic)          42.300      1840.75    No  
4          No              No  Month-to-month              Yes           Electronic check          70.700       151.65   Yes  

"""

df.shape

"""
(7043, 21)
"""

df.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object **** 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB

"""

## >>> TotalCharges sayısal bir değişken olmalı

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

df.info()

"""
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7032 non-null   float64 ***
 20  Churn             7043 non-null   int64

"""

df["Churn"].head()

"""
df["Churn"].head()
Out[13]: 
0    0
1    0
2    1
3    0
4    1
Name: Churn, dtype: int64

"""

# GENEL RESİM

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

"""
##################### Shape #####################
(7043, 21)
##################### Types #####################
customerID           object
gender               object
SeniorCitizen         int64
Partner              object
Dependents           object
tenure                int64
PhoneService         object
MultipleLines        object
InternetService      object
OnlineSecurity       object
OnlineBackup         object
DeviceProtection     object
TechSupport          object
StreamingTV          object
StreamingMovies      object
Contract             object
PaperlessBilling     object
PaymentMethod        object
MonthlyCharges      float64
TotalCharges        float64
Churn                 int64
dtype: object
##################### Head #####################
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport  \
0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No   
1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No   
2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No   
3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes   
4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No   
  StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges  Churn  
0          No              No  Month-to-month              Yes           Electronic check          29.850        29.850      0  
1          No              No        One year               No               Mailed check          56.950      1889.500      0  
2          No              No  Month-to-month              Yes               Mailed check          53.850       108.150      1  
3          No              No        One year               No  Bank transfer (automatic)          42.300      1840.750      0  
4          No              No  Month-to-month              Yes           Electronic check          70.700       151.650      1  
##################### Tail #####################
      customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection  \
7038  6840-RESVB    Male              0     Yes        Yes      24          Yes               Yes             DSL            Yes           No              Yes   
7039  2234-XADUH  Female              0     Yes        Yes      72          Yes               Yes     Fiber optic             No          Yes              Yes   
7040  4801-JZAZL  Female              0     Yes        Yes      11           No  No phone service             DSL            Yes           No               No   
7041  8361-LTMKD    Male              1     Yes         No       4          Yes               Yes     Fiber optic             No           No               No   
7042  3186-AJIEK    Male              0      No         No      66          Yes                No     Fiber optic            Yes           No              Yes   
     TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges  Churn  
7038         Yes         Yes             Yes        One year              Yes               Mailed check          84.800      1990.500      0  
7039          No         Yes             Yes        One year              Yes    Credit card (automatic)         103.200      7362.900      0  
7040          No          No              No  Month-to-month              Yes           Electronic check          29.600       346.450      0  
7041          No          No              No  Month-to-month              Yes               Mailed check          74.400       306.600      1  
7042         Yes         Yes             Yes        Two year              Yes  Bank transfer (automatic)         105.650      6844.500      0  
##################### NA #####################
customerID           0
gender               0
SeniorCitizen        0
Partner              0
Dependents           0
tenure               0
PhoneService         0
MultipleLines        0
InternetService      0
OnlineSecurity       0
OnlineBackup         0
DeviceProtection     0
TechSupport          0
StreamingTV          0
StreamingMovies      0
Contract             0
PaperlessBilling     0
PaymentMethod        0
MonthlyCharges       0
TotalCharges        11
Churn                0
dtype: int64
##################### Quantiles #####################
                0.000  0.050    0.500    0.950    0.990    1.000
SeniorCitizen   0.000  0.000    0.000    1.000    1.000    1.000
tenure          0.000  1.000   29.000   72.000   72.000   72.000
MonthlyCharges 18.250 19.650   70.350  107.400  114.729  118.750
TotalCharges   18.800 49.605 1397.475 6923.590 8039.883 8684.800
Churn           0.000  0.000    0.000    1.000    1.000    1.000

"""

# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 7043
Variables: 21
cat_cols: 17
num_cols: 3
cat_but_car: 1
num_but_cat: 2

"""

cat_cols

"""
Out[17]: 
['gender',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod',
 'SeniorCitizen',
 'Churn']

"""
num_cols

"""
['tenure', 'MonthlyCharges', 'TotalCharges']
"""
cat_but_car

"""
['customerID']
"""

# KATEGORİK DEĞİŞKENLERİN ANALİZİ

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

"""
        gender  Ratio
Male      3555 50.476
Female    3488 49.524
##########################################
     Partner  Ratio
No      3641 51.697
Yes     3402 48.303
##########################################
     Dependents  Ratio
No         4933 70.041
Yes        2110 29.959
##########################################
     PhoneService  Ratio
Yes          6361 90.317
No            682  9.683
##########################################
                  MultipleLines  Ratio
No                         3390 48.133
Yes                        2971 42.184
No phone service            682  9.683
##########################################
             InternetService  Ratio
Fiber optic             3096 43.959
DSL                     2421 34.375
No                      1526 21.667
##########################################
                     OnlineSecurity  Ratio
No                             3498 49.666
Yes                            2019 28.667
No internet service            1526 21.667
##########################################
                     OnlineBackup  Ratio
No                           3088 43.845
Yes                          2429 34.488
No internet service          1526 21.667
##########################################
                     DeviceProtection  Ratio
No                               3095 43.944
Yes                              2422 34.389
No internet service              1526 21.667
##########################################
                     TechSupport  Ratio
No                          3473 49.311
Yes                         2044 29.022
No internet service         1526 21.667
##########################################
                     StreamingTV  Ratio
No                          2810 39.898
Yes                         2707 38.435
No internet service         1526 21.667
##########################################
                     StreamingMovies  Ratio
No                              2785 39.543
Yes                             2732 38.790
No internet service             1526 21.667
##########################################
                Contract  Ratio
Month-to-month      3875 55.019
Two year            1695 24.066
One year            1473 20.914
##########################################
     PaperlessBilling  Ratio
Yes              4171 59.222
No               2872 40.778
##########################################
                           PaymentMethod  Ratio
Electronic check                    2365 33.579
Mailed check                        1612 22.888
Bank transfer (automatic)           1544 21.922
Credit card (automatic)             1522 21.610
##########################################
   SeniorCitizen  Ratio
0           5901 83.785
1           1142 16.215
##########################################
   Churn  Ratio
0   5174 73.463
1   1869 26.537
##########################################

"""

# NUMERİK DEĞİŞKENLERİN ANALİZİ

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

"""
count   7043.000
mean      32.371
std       24.559
min        0.000
5%         1.000
10%        2.000
20%        6.000
30%       12.000
40%       20.000
50%       29.000
60%       40.000
70%       50.000
80%       60.000
90%       69.000
95%       72.000
99%       72.000
max       72.000
Name: tenure, dtype: float64
count   7043.000
mean      64.762
std       30.090
min       18.250
5%        19.650
10%       20.050
20%       25.050
30%       45.850
40%       58.830
50%       70.350
60%       79.100
70%       85.500
80%       94.250
90%      102.600
95%      107.400
99%      114.729
max      118.750
Name: MonthlyCharges, dtype: float64
count   7032.000
mean    2283.300
std     2266.771
min       18.800
5%        49.605
10%       84.600
20%      267.070
30%      551.995
40%      944.170
50%     1397.475
60%     2048.950
70%     3141.130
80%     4475.410
90%     5976.640
95%     6923.590
99%     8039.883
max     8684.800
Name: TotalCharges, dtype: float64

"""

# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


"""
       tenure
Churn        
0      37.570
1      17.979
       MonthlyCharges
Churn                
0              61.265
1              74.441
       TotalCharges
Churn              
0          2555.344
1          1531.796

"""

# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


"""
gender
        TARGET_MEAN  Count  Ratio
Female        0.269   3488 49.524
Male          0.262   3555 50.476
Partner
     TARGET_MEAN  Count  Ratio
No         0.330   3641 51.697
Yes        0.197   3402 48.303
Dependents
     TARGET_MEAN  Count  Ratio
No         0.313   4933 70.041
Yes        0.155   2110 29.959
PhoneService
     TARGET_MEAN  Count  Ratio
No         0.249    682  9.683
Yes        0.267   6361 90.317
MultipleLines
                  TARGET_MEAN  Count  Ratio
No                      0.250   3390 48.133
No phone service        0.249    682  9.683
Yes                     0.286   2971 42.184
InternetService
             TARGET_MEAN  Count  Ratio
DSL                0.190   2421 34.375
Fiber optic        0.419   3096 43.959
No                 0.074   1526 21.667
OnlineSecurity
                     TARGET_MEAN  Count  Ratio
No                         0.418   3498 49.666
No internet service        0.074   1526 21.667
Yes                        0.146   2019 28.667
OnlineBackup
                     TARGET_MEAN  Count  Ratio
No                         0.399   3088 43.845
No internet service        0.074   1526 21.667
Yes                        0.215   2429 34.488
DeviceProtection
                     TARGET_MEAN  Count  Ratio
No                         0.391   3095 43.944
No internet service        0.074   1526 21.667
Yes                        0.225   2422 34.389
TechSupport
                     TARGET_MEAN  Count  Ratio
No                         0.416   3473 49.311
No internet service        0.074   1526 21.667
Yes                        0.152   2044 29.022
StreamingTV
                     TARGET_MEAN  Count  Ratio
No                         0.335   2810 39.898
No internet service        0.074   1526 21.667
Yes                        0.301   2707 38.435
StreamingMovies
                     TARGET_MEAN  Count  Ratio
No                         0.337   2785 39.543
No internet service        0.074   1526 21.667
Yes                        0.299   2732 38.790
Contract
                TARGET_MEAN  Count  Ratio
Month-to-month        0.427   3875 55.019
One year              0.113   1473 20.914
Two year              0.028   1695 24.066
PaperlessBilling
     TARGET_MEAN  Count  Ratio
No         0.163   2872 40.778
Yes        0.336   4171 59.222
PaymentMethod
                           TARGET_MEAN  Count  Ratio
Bank transfer (automatic)        0.167   1544 21.922
Credit card (automatic)          0.152   1522 21.610
Electronic check                 0.453   2365 33.579
Mailed check                     0.191   1612 22.888
SeniorCitizen
   TARGET_MEAN  Count  Ratio
0        0.236   5901 83.785
1        0.417   1142 16.215
Churn
   TARGET_MEAN  Count  Ratio
0        0.000   5174 73.463
1        1.000   1869 26.537

"""

# KORELASYON ANALİZİ

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)

df[num_cols].corr()

"""
Out[10]: 
                tenure  MonthlyCharges  TotalCharges
tenure           1.000           0.248         0.826
MonthlyCharges   0.248           1.000         0.651
TotalCharges     0.826           0.651         1.000

"""

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.corrwith(df["Churn"]).sort_values(ascending=False)

"""
Churn             1.000
MonthlyCharges    0.193
SeniorCitizen     0.151
TotalCharges     -0.199
tenure           -0.352
dtype: float64
"""

# EKSİK DEĞER ANALİZİ

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

"""
              n_miss  ratio
TotalCharges      11  0.160

"""

# Eksik Değer Problemi Giderme
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.isnull().sum()

"""
Out[16]: 
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64
"""

# AYKIRI DEĞER ANALİZİ

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
[48]
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

"""
tenure False
MonthlyCharges False
TotalCharges False
"""

# ÖZELLİK ÇIKARIMI

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"
df.head()

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()
df.shape

"""
önce = (7043, 21) 
sonra = (7043, 31)
10 yeni değişken üretildi.

"""

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
önce

Observations: 7043
Variables: 21  
cat_cols: 17
num_cols: 3
cat_but_car: 1
num_but_cat: 2

sonra

Observations: 7043
Variables: 31   31 - 21 = 10
cat_cols: 24   24 - 21 = 3 
num_cols: 6    6 - 3 = 3 
cat_but_car: 1
num_but_cat: 8   8 - 2 = 6

"""

# ENCODING

df.head()

"""
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport  \
0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No   
1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No   
2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No   
3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes   
4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No   
  
  
  StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges  Churn NEW_TENURE_YEAR  NEW_Engaged  NEW_noProt  \
0          No              No  Month-to-month              Yes           Electronic check          29.850        29.850      0        0-1 Year            0           1   
1          No              No        One year               No               Mailed check          56.950      1889.500      0        2-3 Year            1           1   
2          No              No  Month-to-month              Yes               Mailed check          53.850       108.150      1        0-1 Year            0           1   
3          No              No        One year               No  Bank transfer (automatic)          42.300      1840.750      0        3-4 Year            1           1   
4          No              No  Month-to-month              Yes           Electronic check          70.700       151.650      1        0-1 Year            0           1   
  
  
   NEW_Young_Not_Engaged  NEW_TotalServices  NEW_FLAG_ANY_STREAMING  NEW_FLAG_AutoPayment  NEW_AVG_Charges  NEW_Increase  NEW_AVG_Service_Fee  
0                      1                  1                       0                     0           14.925         0.500               14.925  
1                      0                  3                       0                     0           53.986         0.948               14.238  
2                      1                  3                       0                     0           36.050         0.669               13.463  
3                      0                  3                       0                     1           40.016         0.946               10.575  
4                      1                  1                       0                     0           50.550         0.715               35.350  


"""

cat_cols

"""
Out[40]: 
['gender',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod',
 'NEW_TENURE_YEAR',
 'SeniorCitizen',
 'Churn',
 'NEW_Engaged',
 'NEW_noProt',
 'NEW_Young_Not_Engaged',
 'NEW_TotalServices',
 'NEW_FLAG_ANY_STREAMING',
 'NEW_FLAG_AutoPayment']

"""

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# binary_cols  =  ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
önce

Observations: 7043
Variables: 31  
cat_cols: 24   
num_cols: 6   
cat_but_car: 1
num_but_cat: 8   

sonra 

Observations: 7043
Variables: 31
cat_cols: 24
num_cols: 6
cat_but_car: 1
num_but_cat: 13 **** 13 - 8 = 5 değişken arttı.
"""

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

"""
Out[50]: 
   customerID  gender  Partner  Dependents  tenure  PhoneService  PaperlessBilling  MonthlyCharges  TotalCharges  Churn  NEW_TotalServices  NEW_AVG_Charges  \
0  7590-VHVEG       0        1           0       1             0                 1          29.850        29.850      0                  1           14.925   
1  5575-GNVDE       1        0           0      34             1                 0          56.950      1889.500      0                  3           53.986   
2  3668-QPYBK       1        0           0       2             1                 1          53.850       108.150      1                  3           36.050   
3  7795-CFOCW       1        0           0      45             0                 0          42.300      1840.750      0                  3           40.016   
4  9237-HQITU       0        0           0       2             1                 1          70.700       151.650      1                  1           50.550   
   NEW_Increase  NEW_AVG_Service_Fee  MultipleLines_No phone service  MultipleLines_Yes  InternetService_Fiber optic  InternetService_No  \
0         0.500               14.925                               1                  0                            0                   0   
1         0.948               14.238                               0                  0                            0                   0   
2         0.669               13.463                               0                  0                            0                   0   
3         0.946               10.575                               1                  0                            0                   0   
4         0.715               35.350                               0                  0                            1                   0   
   OnlineSecurity_No internet service  OnlineSecurity_Yes  OnlineBackup_No internet service  OnlineBackup_Yes  DeviceProtection_No internet service  \
0                                   0                   0                                 0                 1                                     0   
1                                   0                   1                                 0                 0                                     0   
2                                   0                   1                                 0                 1                                     0   
3                                   0                   1                                 0                 0                                     0   
4                                   0                   0                                 0                 0                                     0   
   DeviceProtection_Yes  TechSupport_No internet service  TechSupport_Yes  StreamingTV_No internet service  StreamingTV_Yes  StreamingMovies_No internet service  \
0                     0                                0                0                                0                0                                    0   
1                     1                                0                0                                0                0                                    0   
2                     0                                0                0                                0                0                                    0   
3                     1                                0                1                                0                0                                    0   
4                     0                                0                0                                0                0                                    0   
   StreamingMovies_Yes  Contract_One year  Contract_Two year  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  \
0                    0                  0                  0                                      0                               1                           0   
1                    0                  1                  0                                      0                               0                           1   
2                    0                  0                  0                                      0                               0                           1   
3                    0                  1                  0                                      0                               0                           0   
4                    0                  0                  0                                      0                               1                           0   
   NEW_TENURE_YEAR_1-2 Year  NEW_TENURE_YEAR_2-3 Year  NEW_TENURE_YEAR_3-4 Year  NEW_TENURE_YEAR_4-5 Year  NEW_TENURE_YEAR_5-6 Year  SeniorCitizen_1  NEW_Engaged_1  \
0                         0                         0                         0                         0                         0                0              0   
1                         0                         1                         0                         0                         0                0              1   
2                         0                         0                         0                         0                         0                0              0   
3                         0                         0                         1                         0                         0                0              1   
4                         0                         0                         0                         0                         0                0              0   
   NEW_noProt_1  NEW_Young_Not_Engaged_1  NEW_FLAG_ANY_STREAMING_1  NEW_FLAG_AutoPayment_1  
0             1                        1                         0                       0  
1             1                        0                         0                       0  
2             1                        1                         0                       0  
3             1                        0                         0                       1  
4             1                        1                         0                       0  

"""

# BASE MODEL KURULUMU

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

"""
########## LR ##########
Accuracy: 0.8033
Auc: 0.8457
Recall: 0.511
Precision: 0.6706
F1: 0.5793
########## KNN ##########
Accuracy: 0.7704
Auc: 0.7544
Recall: 0.4655
Precision: 0.5861
F1: 0.518
########## CART ##########
Accuracy: 0.728
Auc: 0.6574
Recall: 0.5035
Precision: 0.4875
F1: 0.4953
########## RF ##########
Accuracy: 0.7914
Auc: 0.8269
Recall: 0.4998
Precision: 0.6374
F1: 0.5598
########## SVM ##########
Accuracy: 0.7681
Auc: 0.7255
Recall: 0.2579
Precision: 0.6657
F1: 0.3708
########## XGB ##########
Accuracy: 0.8042  *******champion****
Auc: 0.8459
Recall: 0.5228
Precision: 0.6682
F1: 0.586
########## LightGBM ##########
Accuracy: 0.7968
Auc: 0.8352
Recall: 0.526
Precision: 0.6441
F1: 0.5788
########## CatBoost ##########
Accuracy: 0.7974
Auc: 0.8415
Recall: 0.5206
Precision: 0.6477
F1: 0.5771

"""

# MODEL OPTİMİZASYONU

#  Random Forests

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
Fitting 5 folds for each of 180 candidates, totalling 900 fits

0.8401709849253333

# XGBoost

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
Fitting 5 folds for each of 135 candidates, totalling 675 fits

0.8459305698922485

# LightGBM

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
Fitting 5 folds for each of 36 candidates, totalling 180 fits

0.8456037516831986

# CatBoost

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
Fitting 5 folds for each of 8 candidates, totalling 40 fits

0.8469210460675232

>>>>>  Model opimizasyonu sonrası en iyi değer catboost ile sağlanmıştır. 0.8469210460675232  <<<<<<