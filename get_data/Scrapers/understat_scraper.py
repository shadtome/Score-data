import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import json
from tqdm.auto import tqdm
import time
import random




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
            time.sleep(random.randint(0,3))
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
        column = ['game_id','league','season','club_name','home_away','xG','xGA','npxG','npxGA','ppda','ppda_allowed','deep','deep_allowed',
           'scored','missed','xpts','result','date','wins','draws','loses','pts','npxGD']
        column_data = dict()
        for c in column:
            column_data[c] = []

        for league in tqdm(self.leagues,position=0,leave=True,desc='Leagues',colour='green'):
            for year in tqdm(self.years,position=1,leave=False,desc='Years',colour='red'):
                time.sleep(random.randint(0,3))
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
                        column_data['game_id'].append(id)
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
    
class get_game_roster_stats:
    def __init__(self, last_game_index=26661):
        """Grabs each game stats with the individual events and lineup with their individual player stats
        last_game_index: this is the last game index it will grab, this number will keep going up because
        it is constantly being updated. Furthermore, it can not be no less than 82"""
        self.url = 'https://understat.com/match/'
        self.last_game_index = last_game_index
        self.session = self.create_session()  # Create a session with retries
        self.game_data, self.roster_data, self.events_data = self.get_stats()
    
    def create_session(self):
        """Creates a session with retry strategy to handle connection errors."""
        retry_strategy = Retry(
            total=5,  # Number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on specific status codes
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Apply to these HTTP methods
            backoff_factor=1  # Time between retries (exponential backoff)
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get_stats(self):
        event_column = ['id','minute','result','X','Y','xG','player','h_a','player_id','situation','season','shotType',
                        'match_id','h_team','a_team','h_goals','a_goals','date','player_assisted','lastAction']
        roster_column = ['match_id','goals','own_goals','shots','xG','time','player_id','team_id','position','player','h_a','yellow_card'
                         ,'red_card','roster_in','roster_out','key_passes','assists','xA','xGChain','xGBuildup','positionOrder']
        game_column = ['id','fid','h_id','a_id','date','league_id','season','h_goals','a_goals',
                       'team_h','team_a','h_xg','a_xg','h_w','h_d','h_l','league','h_shot','a_shot','h_shotOnTarget',
                       'a_shotOnTarget','h_deep','a_deep','h_ppda','a_ppda']
        
        game_data = dict()
        roster_data = dict()
        event_data = dict()

        for c in event_column:
            event_data[c]=[]
        for c in roster_column:
            roster_data[c]=[]
        for c in game_column:
            game_data[c]=[]
        

        for i in tqdm(range(81,self.last_game_index),leave=True,desc='matches',colour='green'):
            time.sleep(random.randint(0,3))  # Random sleep to avoid being blocked by the server
            try:
                game_page = self.session.get(self.url + f'{i}', timeout=10)  # Timeout added
            except requests.exceptions.RequestException as e:
                print(f"Failed to retrieve data for game {i}: {e}")
                continue  # Skip to the next game if there's a connection issue
            
            if game_page.status_code == 200:
                soup_game = bs(game_page.text, 'html.parser')
                bad_string = soup_game.find_all('script')

                # Get the game stats data
                string = bad_string[1].string
                event_start = string.index("('")+2
                event_end = string.index("')")
                json_event_data = string[event_start:event_end]
                json_event_data = json_event_data.encode('utf8').decode('unicode_escape')
                events = json.loads(json_event_data)
                event_list = events['h'] + events['a']
                
                # Append the event data
                for x in event_list:
                    for c in event_column:
                        event_data[c].append(x[c])

                next_string = string[event_end+2:]
                game_start = next_string.index("('")+2
                game_end = next_string.index("')")

                json_game_data = next_string[game_start:game_end]
                game_stats = json_game_data.encode('utf8').decode('unicode_escape')
                game_stats = json.loads(game_stats)
                
                for c in game_column:
                    if c == 'h_id':
                        game_data[c].append(game_stats['h'])
                    elif c == 'a_id':
                        game_data[c].append(game_stats['a'])
                    else:
                        game_data[c].append(game_stats[c])

                string = bad_string[2].string
                
                roster_start = string.index("('")+2
                roster_end = string.index("')")
                json_roster_stats = string[roster_start:roster_end]
                
                json_roster_stats = json_roster_stats.encode('utf8').decode('unicode_escape')
                roster_stats = json.loads(json_roster_stats)

                for key_r, value_r in roster_stats.items():
                    for k, v in value_r.items():
                        for c in roster_column:
                            if c == 'match_id':
                                roster_data[c].append(i)
                            else:
                                roster_data[c].append(v[c])

                game_page.close()
        
        df_game = pd.DataFrame(game_data)
        df_roster = pd.DataFrame(roster_data)
        df_events = pd.DataFrame(event_data)
        return df_game, df_roster, df_events
            




            

        

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


def get_lineup_stats_game():
    game_lineup = get_game_roster_stats()
    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)
    fd = os.path.join(fd,'understat')
    if os.path.exists(fd)==False:
        os.mkdir(fd)
    lineup_stats_fd = os.path.join(fd,'lineup_stats.csv')
    game_events_fd = os.path.join(fd,'game_events.csv')
    game_stats_fd = os.path.join(fd,'general_game_stats.csv')

    lineup_stats_df = game_lineup.roster_data
    game_events_df = game_lineup.events_data
    game_stats_df = game_lineup.game_data

    lineup_stats_df.to_csv(lineup_stats_fd,index=False)
    game_events_df.to_csv(game_events_fd,index=False)
    game_stats_df.to_csv(game_stats_fd,index=False)
                    
                    



    
            

        


