from sklearn import model_selection
from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np
import json
import math
import talib
from talib.abstract import *
from Config import *

class Dataset:

    def __init__(self):
        print("> Dataset Class Initialized")
        self.feature_names = [] # For feature comparison and visualization

    """
        Stores Raw historic data of a particular coin in JSON format
    """
    def storeRawCoinHistoricData(self, pair, interval, data):
        print("> Storing Raw Historic Data for ", pair)
        for month, values in data.items():
            with open('datasets/' + pair + '/' + str(interval) + '/' + month + '.json', 'w+') as outfile:
                json.dump(values, outfile)

    """
        Loads raw historic data of a particular coin and converts to JSON format
    """
    def loadRawCoinHistoricData(self, pair, interval, month):
        print("> Loading Raw Historic Data for ", pair)

        data = None

        with open('datasets/' + pair + '/' + str(interval) + '/' + month.replace("-", "_") + '.json', ) as json_file:
            data = json.load(json_file)

        return data

    """
        Loads all the coin data, for the selected months, into an ordered array.
        Each row represents a the status of the coin each minute.
    """
    def loadCoinData(self, pair, interval, months):
        print("> Loading data for ", pair)
        ordered_data = []
        for month in months:
            print(">> Loading month: ", month)
            file_path = DATASET_DIR + pair + '/'+ str(interval) + '/' + month.replace("-", "_") + '.json'

            with open(file_path) as json_file:
                raw_data = json.load(json_file)

            for data in raw_data:
                if len(data) != 0:
                    for minute_info in data: # Data stored every minute
                        ordered_data.append(minute_info)

        return ordered_data

    """
        Creates the train/test sets (X,Y) based on a selection of windows and features.
        Raw variable headings: [time, low, high, open, close, volume]
    """
    def createTrainTestSets(self,coin_name, coin_data, period, shuffling = True):
        print("> Creating X,Y sets for ", coin_name)
        
        technical_indicators = []
        predications = []
        prices = []
        
        positive = 0
        negative = 0

        current_index = 128 # Some random value to just have some offset at the beginning
        end_index = len(coin_data) - 10

        average_close_price = 0
        average_volume = 0

        one_hour_samples = int(1 / (period / 60))

        all_as_dictionary = self.dictionary_data_in_period(coin_data, len(coin_data), len(coin_data))
        rsi = RSI(all_as_dictionary)
        macd = MACD(all_as_dictionary)
        adx = ADX(all_as_dictionary)
        adosc = ADOSC(all_as_dictionary)

        while current_index < end_index:
            features = []
            self.feature_names = []
            
            cur_open_price = float(coin_data[current_index][3])
            cur_close_price = float(coin_data[current_index][4])
            
            current_technical_indicators = self.technical_indicators(coin_data, current_index, period)    
            
            # Momentum indicators
            self.feature_names.append("rsi")
            current_technical_indicators.append(rsi[current_index])
            self.feature_names.append("macd")
            current_technical_indicators.append(macd[2][current_index])
            self.feature_names.append("adx")
            current_technical_indicators.append(adx[current_index])

            # Volume indicators
            self.feature_names.append("adosc")
            current_technical_indicators.append(adosc[current_index])            
            
            prices.append([cur_open_price, cur_close_price])

            technical_indicators.append(current_technical_indicators)

            next_period_avg_price = (coin_data[current_index + 1][3] + coin_data[current_index + 1][4]) / 2 
            change_rate = (next_period_avg_price / cur_close_price) - 1
            
            if change_rate > 0:
                predications.append(1)
                positive += 1
            else:
                predications.append(0)
                negative +=1

            current_index += 1

        print("> Finished Creating set - Size: ", len(technical_indicators)," ", len(predications)," P: ", positive," N: ", negative)
        
        technical_indicators = np.array(technical_indicators)
        # import pdb; pdb.set_trace()
        # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        technical_indicators = preprocessing.scale(technical_indicators)

        if shuffling:
            technical_indicators, predications, prices = shuffle(technical_indicators, predications, prices)

        return technical_indicators, np.array(predications), prices

    def technical_indicators(self, samples, index, period):
        features = []
        
        cur_close_price = float(samples[index][4])
        one_hour_samples = int(1 / (period / 60))
        if period == 15 or period == 5:
            avg_price_1_hour = self.avg_price_in_period(samples, index, one_hour_samples)
            self.feature_names.append("avg_1_hour_price_difference")
            features.append(cur_close_price / avg_price_1_hour)

            bollinger_bands = self.bollinger_bands(samples, index, one_hour_samples)
            self.feature_names.append("lower_bollinger_band_rate")
            features.append(cur_close_price / bollinger_bands[0])
            self.feature_names.append("middle_bollinger_band_rate")
            features.append(cur_close_price / bollinger_bands[1])
            self.feature_names.append("high_bollinger_band_rate")
            features.append(cur_close_price / bollinger_bands[2])

            high_price_1_hour = self.high_price_in_period(samples, index, one_hour_samples)
            low_price_1_hour = self.low_price_in_period(samples, index, one_hour_samples)
            self.feature_names.append("1_hour_high_to_current_price_rate")
            features.append(cur_close_price / high_price_1_hour)
            self.feature_names.append("1_hour_low_to_current_price_rate")
            features.append(cur_close_price / low_price_1_hour)
        elif period == 60:
            avg_price_1_day = self.avg_price_in_period(samples, index, one_hour_samples * 24)
            self.feature_names.append("avg_24_hour_price_difference")
            features.append(cur_close_price / avg_price_1_day)

            bollinger_bands = self.bollinger_bands(samples, index, one_hour_samples * 24)
            self.feature_names.append("lower_bollinger_band")
            features.append(bollinger_bands[0])
            self.feature_names.append("middle_bollinger_band")
            features.append(bollinger_bands[1])
            self.feature_names.append("high_bollinger_band")
            features.append(bollinger_bands[2])

            high_price_1_day = self.high_price_in_period(samples, index, one_hour_samples * 24)
            high_price_1_week = self.high_price_in_period(samples, index, (one_hour_samples * 24) * 7)
            high_price_1_month = self.high_price_in_period(samples, index, (one_hour_samples * 24) * 30)
            low_price_1_day = self.low_price_in_period(samples, index, one_hour_samples * 24)
            low_price_1_week = self.low_price_in_period(samples, index, (one_hour_samples * 24) * 7)
            low_price_1_month = self.low_price_in_period(samples, index, (one_hour_samples * 24) * 30)
            self.feature_names.append("1_day_high_to_current_price_rate")
            features.append(cur_close_price / high_price_1_day)
            self.feature_names.append("1_week_high_to_current_price_rate")
            features.append(cur_close_price / high_price_1_week)
            self.feature_names.append("1_month_high_to_current_price_rate")
            features.append(cur_close_price / high_price_1_month)
            self.feature_names.append("1_day_low_to_current_price_rate")
            features.append(cur_close_price / low_price_1_day)
            self.feature_names.append("1_week_low_to_current_price_rate")
            features.append(cur_close_price / low_price_1_week)
            self.feature_names.append("1_month_low_to_current_price_rate")
            features.append(cur_close_price / low_price_1_month)

        return features

    # One period is time between two samples in used datasets
    def avg_price_in_period(self, samples, start_index, period):
        avg = 0
        index = start_index - period if start_index > period else 0
        for sample in samples[index:start_index]:
            avg += sample[4]

        return avg / len(samples[index:start_index])

    def high_price_in_period(self, samples, start_index, period):
        prices = []
        index = start_index - period if start_index > period else 0
       
        for sample in samples[index:start_index]:
            prices.append(sample[4])

        return np.max(prices)
        
    def low_price_in_period(self, samples, start_index, period):
        prices = []
        index = start_index - period if start_index > period else 0
        
        for sample in samples[index:start_index]:
            prices.append(sample[4])

        return np.min(prices)

    def bollinger_bands(self, samples, start_index, period):
        avg = self.avg_price_in_period(samples, start_index, period) 
        
        avg_bands = 0
        index = start_index - period if start_index > period else 0
        for sample in samples[index:start_index]:
            avg_bands += (sample[4] - avg) ** 2
        
        avg_bands /= len(samples[index:start_index])
        bands_squared = avg_bands ** 2

        upper_band = avg + (2 * bands_squared)
        middle_band = avg
        lower_band = avg - (2 * bands_squared)

        return [lower_band, middle_band, upper_band]

    def dictionary_data_in_period(self, samples, start_index, period):
        inputs = {
            'open': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'close': np.array([]),
            'volume': np.array([])
        }

        index = start_index - period if start_index > period else 0        
        for sample in samples[index:start_index]:
            inputs['low'] = np.append(inputs['low'], sample[1])
            inputs['high'] = np.append(inputs['high'], sample[2])
            inputs['open'] = np.append(inputs['open'], sample[3])
            inputs['close'] = np.append(inputs['close'], sample[4])
            inputs['volume'] = np.append(inputs['volume'], sample[5])
            
        return inputs
