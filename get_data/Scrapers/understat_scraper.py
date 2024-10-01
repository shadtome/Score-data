import requests
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import json
from tqdm.auto import tqdm
import time



class get_player_stats:
    def __init__(self):
        self.url = 'https://understat.com/player/'
        self.data = self.get_stats()

    def get_stats(self):
        player_id = 1
        page = requests.get(self.url + f'/{player_id}')

        # The columns for the dataframe
        columns = ['player_id','name','season','position','games','goals','shots','time','xG','assists','xA','key_passes',
                       'team','yellow','red','npg','npxG','xGChain','xGBuildup']
        #dict object to hold the data
        # and initiate a empty dict
        column_data = dict()
        for c in columns:
                column_data[c]=[]

        
        for pl in tqdm(range(13114),leave=False,desc='main loop',position=0):
            if page.status_code!=404:
                #Get the page html for the player
                player_soup = bs(page.text,'html.parser')

                #First, lets get the players name
                soup_name = player_soup.find_all('meta')
                content = soup_name[2]['content']
                player_name = content.split(',',1)[0]
                
                #Get the script data containing the json string
                # Its really messy but we can clean it up
                bad_script = player_soup.find_all('script')
                string = bad_script[1].string

                #Clean up the string so it is in json format
                ind_start = string.index("('")+2
                ind_end = string.index("')")
                json_data = string[ind_start:ind_end]
                json_data = json_data.encode('utf8').decode('unicode_escape')

                data = json.loads(json_data)
                data = data['season']

                for d in data:
                    column_data['name'].append(player_name)
                    column_data['player_id'].append(player_id)
                    for c in columns[2:]:
                            column_data[c].append(d[c])
            player_id+=1
            page.close()
            page = requests.get(self.url + f'/{player_id}')
            
        
        page.close()
        return pd.DataFrame(column_data)


class get_game_stats:
    def __init__(self):
        self.url = 'https://understat.com/league/'
        self.leagues = ['EPL','La liga','Bundesliga','Serie A','Ligue 1','RFPL']
        self.years = ['2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']
        self.data = self.get_stats()

    def get_stats(self):
        column = ['id','league','season','club_name','home_away','xG','xGA','npxG','npxGA','ppda','ppda_allowed','deep','deep_allowed',
           'scored','missed','xpts','result','date','wins','draws','loses','pts','npxGD']
        column_data = dict()
        for c in column:
            column_data[c] = []

        for league in tqdm(self.leagues,position=0,leave=True,desc='Leagues',colour='green'):
            for year in tqdm(self.years,position=1,leave=False,desc='Years',colour='red'):
                page = requests.get(self.url + f'{league}/{year}')
                game_soup = bs(page.text,'html.parser')

                bad_string = game_soup.find_all('script')

                string = bad_string[2].string

                ind_start = string.index("('")+2
                ind_end = string.index("')")
                json_data = string[ind_start:ind_end]
                json_data = json_data.encode('utf8').decode('unicode_escape')

                data = json.loads(json_data)
                
                for id, val in tqdm(data.items(),position=2,leave=False,desc='Games'):
                    history = val['history']
                    for g in history:
                        column_data['id'].append(id)
                        column_data['club_name'].append(val['title'])
                        column_data['season'].append(year)
                        column_data['league'].append(league)
                        column_data['home_away'].append(g['h_a'])
                        for c in column[5:]:
                            if c=='ppda' or c=='ppda_allowed':
                                if g[c]['def']==0:
                                    column_data[c].append(0)
                                else:
                                    column_data[c].append(g[c]['att']/g[c]['def'])
                            else:
                                column_data[c].append(g[c])
                page.close()
        return pd.DataFrame(column_data)


def player_stats():
    player_stats = get_player_stats()

    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'understat')
    if os.path.exists(fd)==False:
        os.mkdir(fd)
    fd = os.path.join(fd,'player_stats.csv')
    df = player_stats.data
    df.to_csv(fd,index=False)


def game_stats():
    game_stats = get_game_stats()

    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'understat')
    if os.path.exists(fd)==False:
        os.mkdir(fd)
    fd = os.path.join(fd,'game_stats.csv')
    df = game_stats.data
    df.to_csv(fd,index=False)
                    
                    



    
            

        


