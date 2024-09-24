
import get_data.Scrapers.get_kaggle as gk
import get_data.clean_data as clean_data


def get_all_cur_data():
    """Grab all the current data we need"""

    #Grab the data
    gk.get_European_soccer_data()
    gk.get_transfermarkt_data()


    #Do any cleaning needed
    clean_data.transfer_market_to_sql()


    