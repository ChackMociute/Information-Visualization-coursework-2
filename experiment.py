import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from time import sleep, time
from abc import ABC, abstractmethod
from random import choice


class Experiment(ABC):
    NAME = None
    COUNTRY_NAMES = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).name
    def __init__(self, n=10):
        self.path = 'results' if self.NAME is None else os.path.join('results', self.NAME)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.n = n
        self.name = self.get_name()
        self.ans, self.correct, self.times = *np.empty((3, self.n)),
        self.data = pd.DataFrame()
    
    def get_name(self):
        data = os.listdir(self.path)
        start = 1 if len(data) > 0 and data[0] == '.ipynb_checkpoints' else 0
        for i, d in enumerate(data[start:]):
            if f"{i}.json" != d:
                return str(i) + '.json'
        return str(len(data[start:])) + '.json'
    
    def save_data(self):
        with open(os.path.join(self.path, self.name), 'w') as f:
            json.dump({'correct': (self.correct == self.ans).sum()/self.n,
                       'res_mean': self.times.mean(),
                       'res_std': self.times.std()}, f)
            
    def print_results(self):
        print("{0:40}{1:.2}".format('Precentage of correct answers:', (self.correct == self.ans).sum()/self.n))
        print("{0:40}{1:.2}".format('Mean respnse time:', self.times.mean()))
        print("{0:40}{1:.2}".format('Response time standard deviation:', self.times.std()))
        
    def user_input(self, sample):
        print("Which country pollutes the most?")
        for i, c in enumerate(sample):
            print("{0:40}{1}".format(c + ':', i+1))
        s = time()
        return int(input("")), (time() - s)
    
    @abstractmethod
    def run(self):
        pass
    

class BarExperiment(Experiment):
    NAME = 'bar'
    def generate_data(self):
        pollution = pd.DataFrame(np.random.uniform(0, 100, (20, len(self.COUNTRY_NAMES))),
                                 columns=self.COUNTRY_NAMES,
                                 index=reversed(pd.date_range(start='2000', end='2020', freq='Y')))
        self.data = pollution.sum().sort_values(ascending=False).iloc[:20]
        
    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.bar(self.data.index, self.data)
        plt.xticks(rotation=90)
        plt.show()
        
    def run(self):
        for i in range(self.n):
            self.generate_data()
            self.plot()
            answer_sample = pd.concat([self.data.iloc[:1], self.data[1:].sample(4)]).sample(frac=1).index
            self.correct[i] = answer_sample.get_loc(self.data.iloc[:1].index[0]) + 1
            self.ans[i], self.times[i] = self.user_input(answer_sample)
            sleep(1)


class MapExperiment(Experiment):
    NAME = 'map'
    DROP = pd.Index(['Fiji', 'Russia', 'France'])
    CONTINENTS = ['Africa', 'North America', 'Asia', 'Oceania', 'South America', 'Europe']
    def generate_data(self, world):
        pollution = pd.DataFrame(np.random.uniform(0, 100, (20, len(self.COUNTRY_NAMES))),
                                 columns=self.COUNTRY_NAMES,
                                 index=reversed(pd.date_range(start='2000', end='2020', freq='Y')))
        self.data = world.join(pollution.sum().rename('pollution'), on='name').drop(self.DROP)
        
    def plot(self, sample):
        self.data.plot(column='pollution', cmap="Reds", legend=True)
        for label, point in self.data.loc[sample].point.items():
            plt.annotate(sample.get_loc(label) + 1, xy=point, horizontalalignment='center')
        plt.title('Pollution')
        plt.axis('off')
        plt.show()
        
    def run(self):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).set_index('name')
        world['point'] = world['geometry'].apply(lambda x: x.representative_point().coords[:][0])
        for i in range(self.n):
            self.generate_data(world)
            self.data = self.data[self.data["continent"] == choice(self.CONTINENTS)].sort_values('pollution', ascending=False)
            answer_sample = pd.concat([self.data.iloc[:1], self.data[1:].sample(4)]).sample(frac=1).index
            self.correct[i] = answer_sample.get_loc(self.data.iloc[:1].index[0]) + 1
            self.plot(answer_sample)
            self.ans[i], self.times[i] = self.user_input(answer_sample)
            sleep(1)