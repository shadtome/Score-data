
import get_data.Scrapers.get_kaggle as gk
import get_data.Scrapers.understat_scraper as uss
import get_data.clean_data as clean_data


def get_all_kaggle_data():
    """Grab all the current data we need"""

    #Grab the data
    gk.get_European_soccer_data()
    gk.get_transfermarkt_data()
    


    #Do any cleaning needed
    clean_data.transfer_market_to_sql()


def get_understat_data():
    """Get the understat data about players and clubs
    This can take some time for the player data"""

    #grab player stats
    uss.player_stats()
    uss.game_stats()


    #Combine together to make a .db file
    clean_data.understats_to_sql()

    