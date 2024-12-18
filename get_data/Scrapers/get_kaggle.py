
import os
import kaggle



def get_transfermarkt_data():
    """This extracts the transfermarkt data on Kaggle: https://www.kaggle.com/datasets/davidcariboo/player-scores?select=player_valuations.csv
    Note that this kaggle dataset gets update once a week."""
    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'transfermarket')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(dataset='davidcariboo/player-scores',path=fd,unzip=True)

def get_European_soccer_data():
    """This extracts the European Soccer Database data on Kaggle: https://www.kaggle.com/datasets/hugomathien/soccer?resource=download
    This has data from 2008 to 2016.  So not relevant, but might be useful."""
    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'European_data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(dataset='hugomathien/soccer',path=fd,unzip=True)  


def get_understat_data():
    """This is the scraped data from the Understat.com website.  It contains the players stats 
    and the game stats"""

    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'understat')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd,'understat_season_data')

    kaggle.api.authenticate()
    
    kaggle.api.dataset_download_files(dataset='codytipton/understat-data',path=fd,unzip=True)

def get_understat_lineup_data():
    """This is the data that was scraped from the understat.co website, which contains
    more detailed information, in particular, it contains the individual stats per player in each match they played
    in."""
    fd = os.getcwd()
    fd = os.path.join(fd,'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'understat')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd,'understat_game_data')

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(dataset='codytipton/player-stats-per-game-understat',path=fd,unzip=True)



def get_sofascore_data():
    """This is our huge database from Sofascore"""

    fd = os.getcwd()
    fd = os.path.join(fd, 'data')
    if os.path.exists(fd)==False:
        os.mkdir(fd)

    fd = os.path.join(fd, 'sofascore')
    if os.path.exists(fd)==False:
        os.mkdir(fd)
    
    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(dataset = 'rafaelmiksianmagaldi/sofascore-data',path = fd, unzip = True)

