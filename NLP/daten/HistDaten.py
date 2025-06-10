#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datetime import date
import matplotlib.pyplot as plt



def merge_data(df1, df2, save_allData = False):
	df = df1.append(df2, ignore_index=True)

	for cols in df:
	    df[cols] = df[cols].astype(str).str.replace(r"\s+"," ",regex=True)

	df.amount = df.amount.astype(float)

	df.source = (
	    df.source
	    .str.upper()
	    .str.replace(r"BBVA\sKARINA","BBVA",regex="True")
	    .str.replace(r"BBVA\sCARLOS","BBVA",regex="True")
	    .str.replace(r"BBVA\s\w+","BBVA cr",regex="True")
	    .str.replace("REV","Revolut") 
	    .str.replace("RevolutOLUT","Revolut")
	    .str.replace("CASH","cash")
	)
	    
	df.transactiondate = pd.to_datetime(df.transactiondate)
	df.sort_values("transactiondate",inplace=True)

	df.category = df.category.str.replace("nan","Unknown")

	df = df[~df.vendor.isin(['CORTE'])]


	if save_allData == True:
	    today = date.today().strftime("%d-%b-%Y_%s")
	    df.to_csv(f'output/all_data_{today}.csv',index=False,encoding='utf-8')
	    df.to_excel(f'output/all_data_{today}.xlsx',index=False,encoding='utf-8')

	return df


def print_spendInfo(df, currency, who = None):
	df = df[df.currency == currency]
	if who != None:
		df = df[df.who == who.title()]
		print(f"{who}'s Data")
	non_esential = df[~df.category.isin(['Esencial','Maestria'])]
	non_esential_outs = non_esential[non_esential.type.isin(['out'])].amount.sum()

	total_earned = df[df.type.isin(['inf'])].amount.sum()

	amount = round(df.amount.sum(),2)
	print(f"CURRENT AMOUNT: {currency.lower()} "+"{:,.2f}".format(amount))
	print(f"NON-ESSENTIAL SPEND, PERCENTAGE: {-1*round(non_esential_outs/total_earned*100,2)}%")
	print(f"SAVINGS, PERCENTAGE: {round((df.amount.sum())/total_earned*100,2)}%")


def category(df, currency, plot = False, colour = 'k'):
	data = df[df.currency == currency]
	cats = data.groupby('category').sum()

	if plot == True:
		plt.figure(figsize=(7,5))
		plt.bar(cats.index,cats.amount, color = colour)
		plt.xticks(range(len(cats.index)), cats.index, rotation=90)

		plt.title(f"{currency.upper()} Money\n all time")
		plt.show()
	return cats.sort_values('amount',ascending=False)
