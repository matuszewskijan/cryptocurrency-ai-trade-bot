class VirtualAccount:

    def __init__(self):
        self.usd_balance = 1000
        self.btc_amount = 0
        self.btc_balance = 0
        self.btc_price = 0
        self.transactions = []
        self.bought_btc_price = 0
        self.bought_btc_units = 0
        self.last_transaction_was_sell = False;