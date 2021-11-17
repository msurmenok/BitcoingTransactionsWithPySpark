"""
Project for CS259, Fall 2021 by Mariia Surmenok

This script is written for PySpark (Spark version 3.3.1) to do basic analysis for Bitcoin transactions.
The dataset is downloaded from public bitcoin dataset on Google Big Query 'bigquery-public-data.crypto_bitcoin.transactions'

This Python script calculates the following metrics:
    - the average amount of all the transactions
    - he average fee for the transaction per day (also created bar graph to visualize how it changed over time)
    - the total amount of transactions made by each user
    - current worth of user (sent, received, total)
"""

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


# sc = SparkContext("local", "Collect app")
def main():
    conf = SparkConf().setMaster("local[*]").setAppName("Books")
    number_of_cores_used = "8"  # number of cores may be changed to match the system
    conf.set('spark.executor.cores', number_of_cores_used)
    conf.set("spark.default.parallelism", number_of_cores_used)
    spark = SparkSession.builder.appName("BitCoinTransactions").config(conf=conf).getOrCreate()
    sc = spark.sparkContext.getOrCreate()
    sc.setLogLevel("OFF")

    filepath = "file:///Users/msurmenok/Dropbox/Fall2021/project/"
    filename = "bitcoin-transactions.json"
    filename2 = "bitcoin-transactions_small.json"  # file for testing that result is correct

    # load transactions data
    df = spark.read.json(filepath + filename2)

    # show schema
    df.printSchema()

    # show number of partitions and number of rows
    print("number of partitions", df.rdd.getNumPartitions())
    print("number of rows in the original dataset", df.count())

    # calculate average transaction value (transaction value - fee)
    avg_transaction_value = avg_amount_of_all_transactions(df)
    print("Average transaction value: ", avg_transaction_value.collect()[0])

    # calculate average fee per day
    avg_fee = avg_transaction_fee(df)
    print("Average fee per day based on this dataset:")
    avg_fee.show()

    # plot the timeseries result for average fee calculated avove
    show_plot(avg_fee, 'bar', 'date', 'avg', 'Date', 'Average transaction fee, Satoshi')

    # calculate number transactions for each account
    input_address_to_value_df = create_input_value_df(df)
    input_address_to_value_df.show()
    input_address_to_value_df.printSchema()
    print(input_address_to_value_df.count())

    # calculate the total number of transactions (how many times this account sent money) made by each address
    address_to_count_df = get_number_of_transactions_per_user(input_address_to_value_df)
    address_to_count_df.sort(desc('number_transactions')).show()  # sort to show the most active senders

    # calculate current worth of each user based on input and output values
    # input - user sent money, ouput - user received money
    # first, calculate the amount each user sent
    users_sent_df = get_amount_of_money_sent_per_user(input_address_to_value_df)
    users_sent_df.show()

    # create output value df
    output_address_to_value_df = create_output_value_df(df)
    users_received_df = get_amount_of_money_received_per_user(output_address_to_value_df)
    users_received_df.show()
    print(users_received_df.count())

    # join information about accounts that sent money and accounts that received money
    # we use outer join because some account may only sent or only received money in this transaction history
    # and we want to include all of them
    sent_receive_amount_by_user_df = users_sent_df \
        .join(users_received_df, users_sent_df['address'] == users_received_df['address'], 'fullouter') \
        .select(coalesce(users_sent_df['address'], users_received_df['address']), users_sent_df['sent'],
                users_received_df['received']) \
        .na.fill(value=0)  # coalesce used to join together two 'address' columns, na.fill used to replace None with 0

    sent_receive_amount_by_user_df.show()
    print(sent_receive_amount_by_user_df.count())
    # calculate the total balance for each user, received - sent
    user_worth_df = sent_receive_amount_by_user_df.withColumn('total', sent_receive_amount_by_user_df['received'] -
                                                              sent_receive_amount_by_user_df['sent'])
    user_worth_df.sort(desc(user_worth_df['total'])).show()
    print(user_worth_df.count())


def get_amount_of_money_received_per_user(df):
    """
    For each account calculate the amount of money received
    :param df: dataframe with 'address' and 'value' for each input transaction
    :return: dataframe with total value received for each account with columns 'address' and 'received'
    """
    address_to_value_rdd = df.rdd \
        .map(lambda x: (x['address'], x['value'])) \
        .reduceByKey(lambda x1, x2: x1 + x2)
    return address_to_value_rdd.toDF(['address', 'received'])


def get_amount_of_money_sent_per_user(df):
    """
    For each account calculate the amount of money sent
    :param df: dataframe with 'address' and 'value' for each input transaction
    :return: dataframe with total value sent for each account with columns 'address' and 'sent'
    """
    address_to_value_rdd = df.rdd \
        .map(lambda x: (x['address'], x['value'])) \
        .reduceByKey(lambda x1, x2: x1 + x2)
    return address_to_value_rdd.toDF(['address', 'sent'])


def create_input_value_df(df):
    """
    Extract accounts and value that were send from these account from nested structure
    :param df: dataframe of original unchanged bitcoin transactions
    :return: dataframe with address and value columns
    """
    # we have array of inputs, each input is a structure with array of addresses inside
    # since we have two arrays, need to 'explode' twice
    df_exploded = df.select('block_number', explode('inputs').alias('input'))
    df_exploded2 = df_exploded.select('block_number', 'input.value', explode('input.addresses').alias('address'))
    return df_exploded2


def create_output_value_df(df):
    """
    Extract accounts and value that were received by this account from nested structure
    :param df: dataframe of original unchanged bitcoin transactions
    :return: dataframe with address and value received columns
    """
    # we have array of inputs, each input is a structure with array of addresses inside
    # since we have two arrays, need to 'explode' twice
    df_exploded = df.select('block_number', explode('outputs').alias('output'))
    df_exploded2 = df_exploded.select('block_number', 'output.value', explode('output.addresses').alias('address'))
    return df_exploded2


def get_number_of_transactions_per_user(df):
    """
    Calculates number of transactions for each address (account) using rdd methods
    :param df: dataframe with 'address' for each transaction
    :return: dataframe with two columns where first is account address and second is number of transactions for this account
    """
    address_to_count_rdd = df.rdd \
        .map(lambda x: (x['address'], 1)) \
        .reduceByKey(lambda x1, x2: x1 + x2)
    return address_to_count_rdd.toDF(['address', 'number_transactions'])


def avg_amount_of_all_transactions(df):
    """
    Calculate average transaction value excluding fee for transaction
    :param df: dataframe with transactions
    :return: new dataframe with a single average value
    """
    return df.agg({'output_value': 'avg'})


def avg_transaction_fee(df):
    """
    Calculate average transaction fee by day
    :param df: dataframe to plot
    :return: None
    """
    feature_group = ['block_timestamp']
    df = df.withColumn('date', to_date(df['block_timestamp']))
    df_averages = df.groupby('date').agg(avg('fee').alias('avg'))
    return df_averages.sort('date')


def show_plot(df, kind, x, y, x_label, y_label):
    """
    Helper function to draw a plot for given dataframe
    :param df: dataframe to plot
    :param kind: type of chart, such as bar chart
    :param x: x value for plot
    :param y: y value for plot
    :param x_label: x label for plot
    :param y_label: y label for plot
    :return: None
    """
    pddf = df.toPandas()
    pddf.plot(kind=kind, x='date', y='avg')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
