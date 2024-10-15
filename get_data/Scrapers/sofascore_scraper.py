from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm.auto import tqdm
import http.client, json
from urllib.parse import urlparse
import requests
import pandas as pd
import os






class sofascore:
    def __init__(self,country,league,league_id,season_id):
        self.country = country
        self.league = league
        self.season_id = season_id
        self.league_id = league_id
        self.driver = self.initialize_driver()
        self.match_ids,round_clicks = self.scrape_matches()
        

    # Initialize the WebDriver with custom options
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
            print(f"Scrolled to position: {scroll_position}")
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
            print(f"Round {round_number}: Match links located (count: {len(match_links)}).")

            if len(match_links) < 10:
                print(f"Warning: Less than 10 matches found for round {round_number}, possible loading issue.")

            for match in match_links:
                match_url = match.get_attribute('href')
                match_id = match_url.split('#id:')[-1]
                print(match_id)
                if match_id not in match_ids:
                    match_ids.append(match_id)
                    print(f"Found match: {match_url} with Match ID: {match_id}")
        except Exception as e:
            print(f"Error occurred during match collection: {e}")
    
    def find_and_click_left_arrow(self):
        """Find and click the left arrow button to navigate to the previous round."""
        try:
            left_arrow = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "button.Button.iCnTrv[style*='visibility: visible;']"))
            )
            if left_arrow.is_displayed() and left_arrow.is_enabled():
                self.driver.execute_script("arguments[0].click();", left_arrow)
                print("Clicked on the left arrow button.")
                return True
        except Exception as e:
            print(f"Failed to click the left arrow button: {e}")
        return False

    def scrape_matches(self):
        """Main function to scrape match IDs from the Premier League page."""
        match_ids = []
        total_rounds = 38  # Number of rounds in the Premier League
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
                print(f"Total round navigations (left arrow clicks): {round_click_count}")

                # Wait briefly to allow the page to load partially
                time.sleep(3)  # Wait time increased to 3 seconds to allow more content to load

                # Collect match IDs for the current round
                self.collect_match_ids(match_ids, round_click_count + 1)
                
            except Exception as e:
                print(f"An error occurred during scraping: {e}")
                break
        self.driver.close()
        return match_ids, round_click_count
    
    
        

class get_all:
    def __init__(self):
        self.id_dict = {'england': [{'premier-league': {17: [52186, 41886]}}],
                        'spain' : [{'laliga' : {8: [52376,42409]}}]}
        self.match_ids = self.get_match_ids()
        self.data = self.get_data()


    def get_data(self):
        columns = ['name','shortName','position','height','dateOfBirthTimestamp','minutesPlayed','rating'
                   'accuratePass','totalLongBalls','accurateLongBalls','keyPass','totalPass','totalCross','accurateCross'
                   'goalAssist',
                   'savedShotsFromInsideTheBox','saves','totalKeeperSweeper','accurateKeeperSweeper', 
                   'goalsPrevented',
                   'touches','possessionLostCtrl','dispossessed','expectedAssists'
                   'aerialLost','aerialWon','duelLost','duelWon','challengeLost','outfielderBlock'
                   'totalContest','interceptionWon','totalContest','wonContest','totalTackle','interceptionWon',
                   'totalClearance',
                   'blockedScoringAttempt','hitWoodwork','bigChanceCreated','bigChanceMissed',
                   'shotOffTarget','onTargetScoringAttempt','blockedScoringAttempt','goals','expectedGoals',
                   'wasFouled','fouls','totalOffside',]

        data = dict()
        for c in columns:
            data[c] = []

        for id_x in self.match_ids:
            url = "https://www.sofascore.com/api/v1/event/"+id_x+"/lineups"
            parsed_url = urlparse(url)
            conn = http.client.HTTPSConnection(parsed_url.netloc)
            conn.request("GET",parsed_url.path)
            res = conn.getresponse()
            data = res.read()
            jsondata = json.loads(data.decode("utf-8"))
            
            for x,y in jsondata.items():
                if x == 'home' or x == 'away':
                    for player in y[x]['players']:
                        for column, stat in player:
                            if column != 'statistics' and column in columns:
                                data[column].append(stat)
                            elif column == 'statistics':
                                for c in columns:
                                    if c not in player[column].keys():
                                        data[c].append(0)
                                    else:
                                        data[c].append(player[column][c])
            data_df = pd.DataFrame(data)
            conn.close()
            return data_df
    
    def get_match_ids(self):
        match_ids = []
        for country,v in self.id_dict.items():
            for league in v:
                for league_name, data in league.items():
                    for league_id,seasons in data.items():
                        for season_id in seasons:
                            match_ids += sofascore(country,league_name,league_id,season_id).match_ids
        return match_ids

        
    def save_data(self):
        path = os.getcwd()
        path = os.path.join(path,'data/sofascore')
        if os.path.exists(path)==False:
            os.mkdir(path)
        path = os.path.join(path,'sofascore_data.csv')
        self.data.to_csv(path,index=False)
        
          
        
    
       

    
    


