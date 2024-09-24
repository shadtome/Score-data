
import requests
from bs4 import BeautifulSoup as bs
import os








def get_all_premier_league_team_game_data():
    """This extracts the Premier League data from the url = https://www.football-data.co.uk/englandm.php
    from 2002/2003 season to 2024/2025 season
    Note that the 2024/2025 season is ongoing so hence will need to be updated"""

    url = "https://www.football-data.co.uk/englandm.php"
    page = requests.get(url)

    soup = bs(page.text,'html')

    csv_file_names = []

    A_href = soup.find_all('a')
    
    # Extract the href names that correspond premier league data.
    for h in A_href:
        csv_fn = h.get('href')
        if csv_fn[0:7]=='mmz4281' and csv_fn[-6:]=='E0.csv':
            csv_file_names.append(csv_fn)

    # Get csv urls

    csv_urls = []
    for csv in csv_file_names:
        csv_link  = requests.compat.urljoin(url,csv)
        csv_urls.append(csv_link)
    
    # Download the files in the data folder

    for link in csv_urls:
        csv_response = requests.get(link)
        name = link[-11:-7]
        
        fd = os.getcwd()
        fd = os.path.join(fd,'data')
        if os.path.exists(fd)==False:
            os.mkdir(fd)
        fd = os.path.join(fd, f'{name}')
        if os.path.exists(fd)==False:
            os.mkdir(fd)
        
        fd = os.path.join(fd, f'Fixtures.csv')

        with open(fd, 'wb') as file:
            file.write(csv_response.content)

    print('Downloaded Premier League Data')



def update_current_season_team(s):
    """Takes a string for the current Premier league season and updates the stats from
         https://www.football-data.co.uk/englandm.php
        str: put in year in the form xxyy, where 20xx and 20yy, where xx<yy. 
             For example, for 2024/2025 season, I would put 2425"""
    
    url = "https://www.football-data.co.uk/englandm.php"
    page = requests.get(url)

    soup = bs(page.text,'html')

    csv_file_name = ''

    A_href = soup.find_all('a')

    for h in A_href:
        csv_fn = h.get('href')
        if csv_fn[0:7]=='mmz4281' and csv_fn[-11:]==f'{s}/E0.csv':
            csv_file_name = csv_fn

    csv_url = requests.compat.urljoin(url,csv_file_name)

    csv_response = requests.get(csv_url)

    fd = os.getcwd()
    fd = os.path.join(fd,'data')

    fd = os.path.join(fd, f'{s}_PL.csv')

    with open(fd, 'wb') as file:
            file.write(csv_response.content)


