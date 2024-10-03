
import get_data.Scrapers.get_kaggle as gk
import get_data.Scrapers.understat_scraper as uss
import get_data.clean_data as clean_data


def get_all_kaggle_data():
    """Grab all the current data we need"""

    #Grab the data
    gk.get_European_soccer_data()
    gk.get_transfermarkt_data()
    gk.get_understat_data()
    gk.get_understat_lineup_data()
    


    #Do any cleaning needed
    clean_data.transfer_market_to_sql()


def get_understat_data():
    """Get the understat data about players and clubs
    This can take some time for the player data.
    This will take 4 to 5 hours, it is better to just use the kaggle dataset 
    I have uploaded.
    Furthermore, it you want updated data, you will need to run this agian, and it will replace the 
    data with all the data plus the new one.  It is a tricky thing to scrape updates."""

    #grab player stats
    uss.player_stats()
    uss.game_stats()


    #Combine together to make a .db file
    clean_data.understats_to_sql()


def get_understat_game_lineup_data():
    """Get games data with the events recorded (mostly shots) and each individual players stats for that game"""

    uss.get_lineup_stats_game()

    clean_data.understats_lineup_to_sql()



    