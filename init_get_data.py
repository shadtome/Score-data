
import get_data.get_all_data as gd
from get_data.merge_data.merge_transfer_understat import merge_data



# get all the current kaggle data we have
gd.get_all_kaggle_data()
print('Done with Kaggle Datasets')
print('Starting Merge!')
# this does some merging of the data we want and gets our main data set
merge_data()
