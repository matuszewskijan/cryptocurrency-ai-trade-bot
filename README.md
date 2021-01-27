# Crypto Trading AI Bot
Original project came from: https://github.com/gdemos01/crypto-trading-ai-bot-basic but I've made some improvements.
It's using simple classification to predict if price will go up or down in future.
It's my very first time with Python and Tensorflow so the code definitely could be improved.
Accuraccy were 54% at high so it's not ideal, with my current knowledge I have no idea how to improve it.

## How to Use
Currently you can use this code to:
 - Create crypto-currency datasets
 - Train the integrated AI to predict near-future changes of crypto (Up/Down)
 - Run trading simulations with the AutoTrader bot.

### Collect Coin Data For a Specific Month
Collects the data using Coinbase's API and stores them in JSON format.

`python Controller.py --collect_coins --start "2018-11-01" --end "2020-06-30" --pair="BTC-USD" --interval=15`

### Train
Trains the AI using historic crypto information.

`python Controller.py --train --months="2018-11,2018-12,2019-01,2019-02,2019-03,2019-04,2019-05,2019-06,2019-07,2019-08,2019-09,2019-10,2019-11,2019-12" --interval=60 --pair="BTC-USD"`

### Trade
Runs the trading simulation from saved model:
`python Controller.py --trade --months "2020-03,2020-04,2020-05,2020-06" --interval=60 --pair="BTC-USD"`


#### Dependencies
Install TA-Lib for technical indicators calculate:
https://artiya4u.medium.com/installing-ta-lib-on-ubuntu-944d8ca24eae