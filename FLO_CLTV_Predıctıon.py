################# FLO #####################


#master_id= eşsiz müşteri numarası
#order_channel= alışveriş yapılan platforma ait hangi kanalın kullnaıldığı (Android,ios,Desktop,Mobile)
#last_order_channel= en son alışverişin yapıldığı kanal
#first_order_data=müşterinin ilk alışveriş tarihi
#last_order_date=müşterinin son alışveriş tarihi
#last_order_date_online=müşterinin online platformda yapğtığı son alışveriş tarihi
#last_order_date_offline= müşterinin offline platformda yapğtığı son alışveriş tarihi
#order_num_total_ever_offline : Müşterinin offline’da yaptığı toplam alışveriş sayısı
#order_num_total_ever_online : Müşterinin online’da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
#story_type:3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.

!pip install lifetimes

import datetime  as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max.columns",500)
pd.set_option("display.width", 500)
pd.set_option("display.float_format",lambda x:"%.4f" %x)
from sklearn.preprocessing import MinMaxScaler

#######################################
#Görev 1: Veriyi Hazırlama
######################################
#Adım 1: flo_data_20K.csv verisini okuyunuz.

df_=pd.read_csv("Datasets/flo_data_20k.csv")

df=df_.copy()
df.head()

#Adım 2: Aykırı değerleri baskılamak için gerekli olan
# outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
#Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız

def outlier_thresholds(dataframe, variable):
    quartile1= dataframe[variable].quantile(0.01)   #genellıkle 0.25 olarak da alıyoruz.
    quartile3 = dataframe[variable].quantile(0.99)    #genellıkle 0.75
    interquartile_range=quartile3-quartile1
    up_limit= quartile3+1.5*interquartile_range
    low_limit=quartile1-1.5*interquartile_range
    return low_limit,up_limit
# 0.25 0.75 olunca datasetını cok az dokunur, elde verı kalmaz, makıne ogrenmede overfıttıng olur
def replace_with_thresholds(dataframe, variable):
    low_limit,up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=round(low_limit)
    dataframe.loc[(dataframe[variable]>up_limit),variable]=round(up_limit)

#Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"
# değişkenlerinin aykırı değerleri varsa baskılayanız.


df.describe().T

for col in df.columns[df.columns.str.contains("total")]:
    replace_with_thresholds(df,col)

df.describe().T

#ıkıncı yol:
#for col in df.columns:
    #if df[col].dtypes==<float
         #replace_with_thresholds(df,col)


#Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan
# alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz

df["total_order"]=df["order_num_total_ever_offline"]+df["order_num_total_ever_online"]

df["total_price"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]

df.describe().T

#Adım 5: Değişken tiplerini inceleyiniz.
# Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

col_names = [col for col in df.columns if 'date' in col]
df[col_names] = df[col_names].apply(pd.to_datetime)

df.dtypes

#îkıncı yol

date_cols= [col for col in df.columns if "date" in col]
for col in date_cols:
    df[col]=pd.to_datetime(df[col])


#Görev 2: CLTV Veri Yapısının Oluşturulması

#Adım 1: Veri setindeki en son alışverişin yapıldığı
# tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()
today_date=dt.datetime(2021,6,1)

#Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency
# ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i
#oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak,
#recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

df["last_order_difference"]=(df["last_order_date"]-df["first_order_date"]).dt.days

cltv_df=df.groupby("master_id").agg({"last_order_difference": lambda last_order_difference:last_order_difference/7,
                                       "first_order_date": lambda first_order_date: (today_date - first_order_date).dt.days / 7,
                                       "total_order":lambda total_order: total_order,
                                       "total_price": lambda total_price: total_price})

cltv_df

#cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
#cltv_df = pd.DataFrame()

####################### groupbye gerek kalmadan da ılerlenebılır####################
#second yol
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7  #astype yerıne dt.dtypes de olabılır
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
#############################################
#recency_cltv ıle rfm'dekı farklı ; cltv de musterının ılk gun son gun arasındakı alsıverıs sıklıgı ıken
# crm'de analysıs tarıhıne gore hesaplıyoruz -- bu onemlı



cltv_df.columns=["recency_cltv_weekly","T_weekly","frequency","monetary_cltv_ag"]

#Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#Adım 1: BG/NBD modelini fit ediniz.

bgf=BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
           cltv_df["recency_cltv_weekly"],
           cltv_df["T_weekly"])

#• 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve
# exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                    cltv_df["frequency"],
                                                                                    cltv_df["recency_cltv_weekly"],
                                                                                    cltv_df["T_weekly"])
cltv_df

cltv_df.drop("expected_purc_3_month",axis=1,inplace=True)  #fazla kolunu sılmek ıcın

#• 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz
# ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"]=cltv_df["exp_sales_3_month"]*2

cltv_df

#Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama
# bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz

ggf=GammaGammaFitter(penalizer_coef=0.01) #

ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_ag"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit (cltv_df["frequency"],cltv_df["monetary_cltv_ag"])

cltv_df

#Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df['cltv']=ggf.customer_lifetime_value(bgf,
                                 cltv_df["recency_cltv_weekly"],
                                 cltv_df["T_weekly"],
                                 cltv_df["monetary_cltv_ag"],
                                 cltv_df["frequency"],
                                 time=6,  #6 aylık
                                 freq="W" , #T nın frekans bılgısı
                                 discount_rate=0.02)
cltv_df


# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv",ascending=False).head(20)


#Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması

#Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi
# 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz


cltv_df["segment"]=pd.qcut(cltv_df['cltv'],4,labels=["D","C","B","A"])

cltv_df

#Adım 2: 4 grup içerisinden seçeceğiniz 2 grup
# için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("segment").agg({"count","mean","sum"})






