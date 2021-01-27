from Dataset import Dataset
from Model import Model
from Config import *
from AutoTrader import AutoTrader
import argparse
from CoinbaseAPI import CoinbaseAPI

if __name__ == '__main__':

    print("\n\n\n                   ,.=ctE55ttt553tzs.,                           \n",
          "              ,,c5;z==!!::::  .::7:==it3>.,                      \n",
          "           ,xC;z!::::::    ::::::::::::!=c33x,                   \n",
          "         ,czz!:::::  ::;;..===:..:::   ::::!ct3.                 \n",
          "       ,C;/.:: :  ;=c!:::::::::::::::..      !tt3.               \n",
          "      /z/.:   :;z!:::::J  :E3.  E:::::::..     !ct3.             \n",
          "    ,E;F   ::;t::::::::J  :E3.  E::.     ::.     \ztL            \n",
          "   ;E7.    :c::::F******   **.  *==c;..    ::     Jttk           \n",
          "  .EJ.    ;::::::L                    \:.   ::.    Jttl          \n",
          "  [:.    :::::::::773.    JE773zs.     I:. ::::.    It3L         \n",
          " ;:[     L:::::::::::L    |t::!::J     |::::::::    :Et3         \n",
          " [:L    !::::::::::::L    |t::;z2F    .Et:::.:::.  ::[13  CRYPTO \n",
          " E:.    !::::::::::::L               =Et::::::::!  ::|13  TRADING \n",
          " E:.    (::::::::::::L    .......       \:::::::!  ::|i3  AI      \n",
          " [:L    !::::      ::L    |3t::::!3.     ]::::::.  ::[13          \n",
          " !:(     .:::::    ::L    |t::::::3L     |:::::; ::::EE3         \n",
          "  E3.    :::::::::;z5.    Jz;;;z=F.     :E:::::.::::II3[         \n",
          "  Jt1.    :::::::[                    ;z5::::;.::::;3t3          \n",
          "   \z1.::::::::::l......   ..   ;.=ct5::::::/.::::;Et3L          \n",
          "    \z3.:::::::::::::::J  :E3.  Et::::::::;!:::::;5E3L           \n",
          "     \cz\.:::::::::::::J   E3.  E:::::::z!     ;Zz37`            \n",
          "       \z3.       ::;:::::::::::::::;='      ./355F              \n",
          "         \z3x.         ::~======='         ,c253F                \n",
          "           \ z3=.                      ..c5t32^                  \n",
          "              =zz3==...          ...=t3z13P^                     \n",
          "                   `*=zjzczIIII3zzztE3>*^`                        \n\n\n")


    print("\n> Welcome to Crypto-Trading AI")

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_coins", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--trade", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
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

        print("> Creating Testing Data for ", COIN_PAIR)
        data = dataset.loadCoinData(COIN_PAIR, TESTING_MONTHS)
        x_test, y_test, prices = dataset.createTrainTestSets(COIN_PAIR, data, training_window=TRAINING_WINDOW, labeling_window=LABELING_WINDOW)

        test_model = Model("AutoTraderAI", x_train)
        test_model.train(x_train, y_train, batch_size=64, epochs=10)
        # test_model.evaluate(x_test,y_test)

        auto_trader = AutoTrader(test_model)
        auto_trader.runSimulation(x_test, prices)
    else:
        print("> The biggest mistake you can make in life is to waste your time. – Jerry Bruckner")
        print("> P.S. Use an argument next time: --collect_coins or --train_and_trade")








