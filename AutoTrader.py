from VirtualAccount import VirtualAccount
import numpy as np
from Config import *
import time

class AutoTrader:

    def __init__(self,model):
        self.advisor = model
        self.account = VirtualAccount()
        self.trade_amount = 100
        self.next_window_price = 0

    def buy(self):
        if self.account.bought_btc_units >= 5: # We should avoid going all-in
            self.sell(1.05) # Unlikely scenario but sell when profit is huge (>5%)
            return

        current_transactions = 1
        while self.account.usd_balance - self.trade_amount >= 0 and self.account.bought_btc_units <= 5 and current_transactions <= 2:
            print(">> BUYING $", self.trade_amount, " WORTH OF BITCOIN")
            transaction = {}
            transaction['btc_amount'] = (self.trade_amount / self.next_window_price) * 0.995
            transaction['btc_price'] = self.next_window_price
            transaction['usd_amount'] = self.trade_amount
            self.account.transactions.append(transaction)

            self.account.btc_amount += (self.trade_amount / self.next_window_price) * 0.995
                self.account.usd_balance -= self.trade_amount
            self.account.bought_btc_price = self.next_window_price
            self.account.bought_btc_units += 1
            current_transactions += 1

    def sell(self):
        if self.account.btc_balance - self.trade_amount >= 0:
            if self.account.btc_price > self.account.bought_btc_at: # Is it profitable?
                print(">> SELLING $",self.trade_amount," WORTH OF BITCOIN")
                self.account.btc_amount -= (self.trade_amount / self.account.btc_price)
                self.account.usd_balance += self.trade_amount
                self.account.last_transaction_was_sell = True
            else:
                print(">> Declining sale: Not profitable to sell BTC")
        else:
            print(">> Not enough BTC left in your account to buy USD ")

    def runSimulation(self, samples, prices, interval, trade_months):
        print("> Trading Automatically for ", trade_months)
        day_count = 0
        one_hour_interval = int(1 / (interval / 60))
        for i in range(0, len(samples) - 1):
            if i % (one_hour_interval * 24) == 0:
                day_count += 1
                print("##########################################   DAY ", day_count ,"   #########################################")
                print("#           Account Balance: $", (self.account.usd_balance + self.account.btc_balance), " BTC: $",
                      self.account.btc_balance, " USD: $", self.account.usd_balance, "")

                prediction = self.advisor.predict(np.array([samples[i]]))
            if prediction == None or prediction == 2:
                self.sell(sell_price=1.03)
                continue

            btc_price = prices[i][0]
            self.next_window_price = prices[i + 1][0]

                if self.account.btc_price != 0:
                    self.account.btc_balance = self.account.btc_amount * btc_price

                self.account.btc_price = btc_price

                if prediction == 1:
                if self.next_window_price / btc_price < 0.99:
                    self.buy()
                elif self.next_window_price / btc_price > 1:
                    self.sell()
                else:
                    self.sell()

                self.account.btc_balance = self.account.btc_amount * btc_price

        print("#################################################################################################")
        print("#           Account Balance: $", (self.account.usd_balance + self.account.btc_balance), " BTC: $",
              self.account.btc_balance, " USD: $", self.account.usd_balance, "")
        print("#################################################################################################")