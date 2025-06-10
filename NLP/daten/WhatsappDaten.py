#!/usr/bin/env python
# coding: utf-8

# In[Read Python libraries and generalities]:

from datetime import date

import pandas as pd
import numpy as np 

cat_map = {
    "Gp":"Gustos Personales",
    "Cp":"Cuidados Personales",
    "Cr":"Credito",
    "Es":"Esencial",
    "Ma":"Maestria",
    "Sal":"Salidas",
    "Fch":"Ch. F. Familia",
    "Chf":"Ch. F. Familia",
    "Inv":"Inversiones",
    "Don":"Donaciones",
    "Fa":"Familia / Amigos",
    "Unk":"Unknown"
}

def read_data(path,filename, month_year):
    # In[Read WhatsApp data]:
    df = pd.read_csv(
        path+filename, 
        header = None, 
        delimiter = "\t", 
        sep=" - ", 
        engine='python'
    ).rename(columns={0:"data"})

    # In[Search for latest register]:
    df["cortes"] = df.data.str.find("OUTFLOW/INFLOW")
    last_refresh_index = df[~df.cortes.isin([-1])].iloc[-1].name
    WA_data = df[last_refresh_index+1:].drop("cortes",axis=1)

    # In[Spit data based on bars]:
    WA_data["bars_count"] = WA_data.data.str.find("|")
    WA_data = WA_data[~WA_data.bars_count.isin([-1])].drop("bars_count",axis=1)
    WA_data.reset_index(drop=True,inplace=True)

    # In[Extract imperative info]:
    WA_data["posteddate"] = WA_data.data.str.extract(pat = r"(\d{1,2}/\d{1,2}/\d{2})")
    WA_data["who"] = WA_data.data.str.extract(pat = r"(Carlos|Karina)")
    WA_data["type"] = WA_data.data.apply(lambda x: x.split("|")[0].strip()[-3:])
    WA_data["lenght"] = WA_data.data.apply(lambda x: len(x.split("|")))
    WA_data.data = WA_data.apply(lambda df: df["data"] + " |" if df["lenght"] == 8 else df["data"],axis=1)
    WA_data.drop("lenght",axis=1,inplace=True)

    # In[Create list of lists]:
    levels_list = WA_data.data.str.split("|").apply(lambda x: x[1:]).to_list()
    flat_list = np.array([item for sublist in levels_list for item in sublist])

    # In[Turn list into a DataFrame]:
    # Reshaping flat list with (n_rows, n_cols) to convert it to df
    cols = ["transactiondate","currency","source","amount","vendor","memo","category","localcurrency"]
    data = pd.DataFrame(np.reshape(flat_list,(len(WA_data), 8)), columns = cols)

    for cols in data:
        data[cols] = data[cols].astype(str).str.strip()

    data.currency = data.currency.str.lower()
    data.category = data.category.str.title()
    data.transactiondate = (
        data.transactiondate
        .str.replace(r"\.21",".2021",regex=True)
        .apply(lambda x: x+month_year if len(x) == 2 else x)
    )

    data.source = (
        data.source
        .str.upper()
        .str.replace(r"BBVA\s\w+","BBVA cr",regex="True")
        .str.replace("REV","Revolut")
        .str.replace("CASH","cash")
    )

    data.category = (
        data.category
        .map(cat_map)
        .fillna(data.category)
        .str.replace("Bbva","BBVA")
    )

    data.amount = data.amount.astype(float)
    data.vendor = data.vendor.str.title()
    data.loc[data.localcurrency.isin([""]),"localcurrency"] = np.nan
    data.localcurrency = data.localcurrency.fillna(data.currency)

    # In[WARNING: this section might change per refresh]:
    # Custom Fix of this refresh
    data.category = data.category.str.replace("Tp","Gustos Personales")
    print("WARNING: Remember to update the custom code for new data in WatsappDaten.py")

    # In[Merging DataFrames]:
    data.insert(loc = 0, column = "type", value = WA_data.type)
    data.insert(loc = len(data.columns)-1, column = "who", value = WA_data.who)
    data.insert(loc = 1, column = "posteddate", value = WA_data.posteddate)

    # In[Normalizing datime format for date-related columns]:
    data.posteddate = pd.to_datetime(data.posteddate,format="%m/%d/%y").dt.date
    data.transactiondate = pd.to_datetime(data.transactiondate,format="%d.%m.%Y").dt.date

    data.sort_values("transactiondate",inplace=True)
    data.reset_index(drop=True,inplace=True)

    data.type = data.type.str.lower().str.replace("inv","inf")

    data.amount = data.amount.astype(float)
    data.amount = data.apply(lambda df: df['amount']*-1 if df['type'] == 'out' else df['amount'],axis=1)

    return data

#read_data(filename="WhatsApp Chat with Transacciones .txt", month_year=".06.2021")

