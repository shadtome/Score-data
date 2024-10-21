import get_data.Scrapers.get_kaggle as gk
import get_data.merge_data.merge_transfer_sofascore as mts

# get all kaggle data
print('Start kaggle data download')
gk.get_transfermarkt_data()
gk.get_sofascore_data()
print('Done with kaggle data\n Start Merge')
mts.merge()