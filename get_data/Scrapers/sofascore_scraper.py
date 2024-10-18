from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm.auto import tqdm
import tqdm as tqdm_m
import http.client, json
from urllib.parse import urlparse
import requests
import pandas as pd
import os
from datetime import datetime
import random






class sofascore:
    def __init__(self,country,league,league_id,season_id,num_rounds):
        self.country = country
        self.league = league
        self.season_id = season_id
        self.league_id = league_id
        self.num_rounds = num_rounds
        self.driver = self.initialize_driver()
        self.match_ids,round_clicks = self.scrape_matches()
        
    def initialize_driver(self):
        chrome_options = ChromeOptions()
        # Set the page load strategy to 'none'
        chrome_options.page_load_strategy = 'none'
        
        return webdriver.Chrome(options=chrome_options)
    
    def navigate_to_page(self,url):
        """Navigate to the given URL without waiting for the full page load."""
        
        self.driver.get(url)
        time.sleep(2)  # Short initial wait to start loading the page

    def scroll_to_matches(self):
        """Scroll down to the Matches section of the page."""
        try:
            # Scroll to the estimated position to bring the Matches section into view
            scroll_position = 1700  # Adjusted value for the Matches section
            self.driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            #print(f"Scrolled to position: {scroll_position}")
        except Exception as e:
            print(f"An error occurred while scrolling: {e}")

    def collect_match_ids(self,match_ids, round_number):
        """Collect unique match IDs from the currently displayed Matches section."""
        try:
            
            # Ensure the elements are fully loaded
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[href*="/football/match/"]'))
            )
            
            # Second scroll to ensure all matches are visible
            self.scroll_to_matches()
            
            time.sleep(1)  # Allow time for full rendering after scroll

            match_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/football/match/"]')
            
            #print(f"Round {round_number}: Match links located (count: {len(match_links)}).")

            if len(match_links) < 10:
                print(f"Warning: Less than 10 matches found for round {round_number}, possible loading issue.")
                

            for match in match_links:
                
                match_url = match.get_attribute('href')
                
                match_id = match_url.split('#id:')[-1]
                
                if match_id not in match_ids:
                    match_ids.append(match_id)
                    #print(f"Found match: {match_url} with Match ID: {match_id}")
        except Exception as e:
            print(f"Error occurred during match collection: {e}")

    def refresh_page(self, retries=3):
        """Refresh the page if match loading issues occur."""
        for attempt in range(retries):
            try:
                self.driver.refresh()
                time.sleep(5)
                return True
            except Exception as e:
                print(f"Error refreshing the page (attempt {attempt+1}): {e}")
                if attempt == retries - 1:
                    return False
                time.sleep(5)
        return False
    
    def find_and_click_left_arrow(self):
        """Find and click the left arrow button to navigate to the previous round."""
        try:
            
            left_arrow = WebDriverWait(self.driver, 10).until(
                #EC.visibility_of_element_located((By.CSS_SELECTOR, "button.Button.iCnTrv[style*='visibility: visible;']"))
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.Button.iCnTrv[style*='visibility: visible;']"))
            )
            
            if left_arrow.is_displayed() and left_arrow.is_enabled():
                self.driver.execute_script("arguments[0].click();", left_arrow)
                #print("Clicked on the left arrow button.")
                return True
        except Exception as e:
            print(f"Failed to click the left arrow button: {e}")
        return False

    def scrape_matches(self):
        """Main function to scrape match IDs from the Premier League page."""
        match_ids = []
        total_rounds = self.num_rounds  # Number of rounds in the league
        round_click_count = 0

        # Navigate to the Premier League page for the desired season
        self.navigate_to_page(f"https://www.sofascore.com/tournament/football/{self.country}/{self.league}/{self.league_id}#id:{self.season_id}")
        self.scroll_to_matches()

        # Collect match IDs for the initial round
        self.collect_match_ids(match_ids, round_click_count + 1)

        while round_click_count < (total_rounds - 1):  # Loop for 37 left arrow clicks
            try:
                
                # Attempt to click the left arrow button
                if not self.find_and_click_left_arrow():
                    print("Left arrow button is no longer interactable. Ending loop.")
                    break
                
                round_click_count += 1
                #print(f"Total round navigations (left arrow clicks): {round_click_count}")

                # Wait briefly to allow the page to load partially
                time.sleep(3)  # Wait time increased to 3 seconds to allow more content to load

                # Collect match IDs for the current round
                self.collect_match_ids(match_ids, round_click_count + 1)
                
            except Exception as e:
                print(f"An error occurred during scraping: {e}")
                #refresh and retry if scraping fails
                if not self.refresh_page():  
                    break
        self.driver.close()
        return match_ids, round_click_count
    
class get_all_ids:
    def __init__(self):
        None
        
    def collect_match_ids(self):
        match_ids = dict()
        id_dict = self.league_season_ids_from_file()

        #id_dict = {"england": {"premier-league": {"league_id": 17,"seasons": {"23/24": {"id":52186,"rounds":38,"matches":380, 'name': 'Premier League'}}}},
                   #"spain" : {"laliga": {"league_id": 8, "seasons": {"23/24": {"id":52376,"rounds":38,"matches":380,"name":"LaLiga"}}}}}
        
        for country,country_v in tqdm(id_dict.items(),leave=False,desc='country',position=0):
            for league, league_v  in tqdm(country_v.items(),leave=False,desc='leagues',position=1):
                for season_year,season_v in tqdm(league_v['seasons'].items(),leave=False,desc='season_ids',position=2):
                            
                            
                            xm = season_v['matches']
                            list_ids = sofascore(country,league,league_v['league_id'],season_v['id'],xm).match_ids
                            
                            match_ids = match_ids | {country : 
                                                            {league: 
                                                                    {season_year: {
                                                                                    "match_ids": list_ids,
                                                                                    "matches": xm,
                                                                                    "name": season_v['name']} }} }
        return match_ids
    
    def list_match_ids(self):
        map_match_ids = self.match_ids_from_file()
        match_ids_list = []
        for country, country_v in map_match_ids.items():
            for league,league_v in country_v.items():
                for season_year,season_v in league_v.items():
                    match_ids_list += season_v['match_ids']
        return match_ids_list
    
    def clean_ids(self):
        match_ids = self.match_ids_from_file()
        for country, country_v in match_ids.items():
            for league,league_v in country_v.items():
                for season_year,season_v in league_v.items():
                    xm = season_v['matches']
                    list_data = season_v['match_ids']
                    name = season_v['name']
                    list_data = self.clean_season_ids(name,season_year,xm,list_data)
                    season_v['match_ids'] = list_data
        path = os.getcwd()
        path = os.path.join(path,'pre_data/sofascore/match_ids.json')      
        with open(path, 'w',encoding='utf-8') as json_file:
            json.dump(match_ids,json_file, ensure_ascii=False,indent=4)
    
    def clean_season_ids(self,xleague,xseason,xm,list_data):
        """Go through the current match ids to make sure they are in the correct season and league."""
        if len(list_data) == xm:
            return list_data
        removed_list = []
        league_list = []
        i=0
        pbar = tqdm(total=len(list_data),ascii="--")
        while i<len(list_data):
            
            time.sleep(random.randint(0,2))
            
            url_date = "https://www.sofascore.com/api/v1/event/"+list_data[i]
            parsed_url_date = urlparse(url_date)
            conn_date = http.client.HTTPSConnection(parsed_url_date.netloc)
            conn_date.request("GET",parsed_url_date.path)
            res_date = conn_date.getresponse()
            date_json = res_date.read()
            json_data_date = json.loads(date_json.decode("utf-8"))
            league = json_data_date['event']['tournament']['name']
            season = json_data_date['event']['season']['year']
            conn_date.close()
            
            # lets compare this with what the actual league and season should be.
            if league != xleague or season != xseason:
                removed_list.append(list_data.pop(i))
                league_list.append(league)
                pbar.update(2)
            else:
                i+=1
                pbar.update(1)

        pbar.close()
        print(f'removed the elements {removed_list}, leaving {len(list_data)} left over\n {league_list}')
        return list_data
            


    def collect_and_save(self):
        match_ids_map = self.collect_match_ids()
        path = os.getcwd()
        path = os.path.join(path,'pre_data/sofascore/match_ids.json')
        with open(path, 'w',encoding='utf-8') as json_file:
            json.dump(match_ids_map,json_file, ensure_ascii=False,indent=4)

    def match_ids_from_file(self):
        path = os.getcwd()
        path = os.path.join(path, 'pre_data/sofascore/match_ids.json')
        with open(path,'r') as json_file:
            match_ids = json.load(json_file)
        return match_ids

    def league_season_ids_from_file(self):
        path = os.getcwd()
        path = os.path.join(path,'pre_data')
        if os.path.exists(path)==False:
            os.mkdir(path)
        path = os.path.join(path,'sofascore')
        if os.path.exists(path)==False:
            os.mkdir(path)
        path = os.path.join(path,'league_seasons.json')
        with open(path,'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict

        

class get_all_data:
    def __init__(self):
        self.match_ids = get_all_ids().list_match_ids()
        self.data = self.get_data()
        self.save_data()

    def get_data(self):
        top_columns = ['name','shortName','position','height','dateOfBirthTimestamp']
        game_stats_columns = ['team','date','league','season']
        columns = ['minutesPlayed','rating'
                   'accuratePass','totalLongBalls','accurateLongBalls','keyPass','totalPass','totalCross','accurateCross'
                   'goalAssist',
                   'savedShotsFromInsideTheBox','saves','totalKeeperSweeper','accurateKeeperSweeper', 
                   'goalsPrevented',
                   'touches','possessionLostCtrl','dispossessed','expectedAssists'
                   'aerialLost','aerialWon','duelLost','duelWon','challengeLost','outfielderBlock'
                   'totalContest','interceptionWon','totalContest','wonContest','totalTackle',
                   'totalClearance',
                   'blockedScoringAttempt','hitWoodwork','bigChanceCreated','bigChanceMissed',
                   'shotOffTarget','onTargetScoringAttempt','goals','expectedGoals',
                   'wasFouled','fouls','totalOffside',]

        data = dict()
        for tc in top_columns:
            data[tc]=[]

        for gc in game_stats_columns:
            data[gc]=[]

        for c in columns:
            data[c] = []
        num_players = 0
        for id_x in tqdm(self.match_ids,leave=False,desc='main loop',position=0):
            time.sleep(random.randint(0,2))
            
            
            url_date = "https://www.sofascore.com/api/v1/event/"+id_x
            parsed_url_date = urlparse(url_date)
            conn_date = http.client.HTTPSConnection(parsed_url_date.netloc)
            conn_date.request("GET",parsed_url_date.path)
            res_date = conn_date.getresponse()
            date_json = res_date.read()
            json_data_date = json.loads(date_json.decode("utf-8"))
            date = datetime.fromtimestamp(json_data_date['event']['startTimestamp'])
            h_team = json_data_date['event']['homeTeam']['name']
            a_team = json_data_date['event']['awayTeam']['name']
            league = json_data_date['event']['tournament']['name']
            season = json_data_date['event']['season']['year']
            conn_date.close()


            #player stats
            url = "https://www.sofascore.com/api/v1/event/"+id_x+"/lineups"
            parsed_url = urlparse(url)
            conn = http.client.HTTPSConnection(parsed_url.netloc)
            conn.request("GET",parsed_url.path)
            res = conn.getresponse()
            data_json = res.read()
            jsondata = json.loads(data_json.decode("utf-8"))

            

            
            
            for x,y in jsondata.items():
                if x == 'home' or x == 'away':
                    for player in y['players']:
                        num_players+=1
                        for tc in top_columns:
                            if tc not in player['player'].keys():
                                data[tc].append(None)
                            elif tc == 'dateOfBirthTimestamp':
                                data[tc].append(datetime.fromtimestamp(player['player'][tc]))
                            else:
                                data[tc].append(player['player'][tc])
                        if x == 'home':
                            data['team'].append(h_team)
                        if x == 'away':
                            data['team'].append(a_team)
                        data['date'].append(date)
                        data['league'].append(league)
                        data['season'].append(season)
                        if 'statistics' in player.keys():
                            for c in columns:
                                
                                if c not in player['statistics'].keys():
                                    data[c].append(0)
                                else:
                                    data[c].append(player['statistics'][c])
                        else:
                            for c in columns:
                                data[c].append(0)
            
            
            conn.close()
        
        data_df = pd.DataFrame(data)
        data_df['dateOfBirthTimestamp'] = pd.to_datetime(data_df['dateOfBirthTimestamp'])
        return data_df
       
    def save_data(self):
        path = os.getcwd()
        path = os.path.join(path,'data/sofascore')
        if os.path.exists(path)==False:
            os.mkdir(path)
        path = os.path.join(path,'sofascore_data.csv')
        self.data.to_csv(path,index=False)
        
          
        
    
       

    
    


