from Dataset import Dataset
from Model import Model
from Config import *
import argparse
import datetime

from AutoTrader import AutoTrader
from CoinbaseAPI import CoinbaseAPI

import matplotlib.pyplot as plt

if __name__ == '__main__':

    print("\n> Welcome to Crypto-Trading AI")

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_coins", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--trade", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--months")
    parser.add_argument("--pair")
    parser.add_argument("--interval")
    args = parser.parse_args()

    dataset = Dataset()

    if args.collect_coins:
        start_date = args.start if args.start else "2020-01-01"
        end_date = args.end if args.end else "2020-02-02"
        pair = args.pair if args.pair else "BTC-USD"        
        interval = int(args.interval) if args.interval else 30

        coinbaseAPI = CoinbaseAPI()
        historic_data = coinbaseAPI.getCoinHistoricData(pair, start=start_date, end=end_date, granularity=interval * 60)
        dataset.storeRawCoinHistoricData(pair, interval, historic_data)

        print("> Using Coinbase API to build dataset for ",COIN_PAIR)
    elif args.train:
        print("> Creating Training Data for ", COIN_PAIR)

        pair = args.pair if args.pair else "BTC-USD"
        interval = int(args.interval) if args.interval else 15

        training_months = args.months.split(',') if args.months else TRAINING_MONTHS

        data = dataset.loadCoinData(pair, interval, training_months)        
        x_train, y_train, _ = dataset.createTrainTestSets(pair, data, interval)

        data = dataset.loadCoinData(pair, interval, ["2020-03", "2020-04", "2020-05", "2020-06"])
        test_tech_indicators, test_predictions, _ = dataset.createTrainTestSets(pair, data, interval)

        test_model = TradeModel("AutoTraderAI", x_train)        
        history = test_model.train(x_train, y_train, test_tech_indicators, test_predictions, batch_size=256, epochs=100)
        test_model.evaluate(test_tech_indicators, test_predictions)
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.show()
        
        test_model.model.save("models/" + pair + "/" + str(interval))
    elif args.trade:
        pair = args.pair if args.pair else "BTC-USD"
        interval = int(args.interval) if args.interval else 15
        trading_months = args.months.split(',') if args.months else TRADING_MONTHS

        data = dataset.loadCoinData(pair, interval, trading_months)
        x_test, y_test, prices = dataset.createTrainTestSets(pair, data, interval, shuffling = False)

        model_path = "models/" + pair + "/" + str(interval)

        model = TradeModel("AutoTraderAI", x_test, model_path)
        auto_trader = AutoTrader(model)
        auto_trader.runSimulation(x_test, prices, interval, trading_months)
    else:
        print("> The biggest mistake you can make in life is to waste your time. – Jerry Bruckner")
        print("> P.S. Use an argument next time: --collect_coins or --train_and_trade")








