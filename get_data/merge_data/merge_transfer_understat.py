import pandas as pd
import os
from tqdm.auto import tqdm
import sqlite3 as sql

class merge_data:
    def __init__(self):
        self.understat_df = self.get_understat()
        self.transfer_df = self.get_transfer()
        self.combine()

    def get_understat(self):
        con = sql.connect('data/understat/understat_game_data/understat_lineup_game_stats.db')

        lineup = pd.read_sql("""SELECT  game.date,
                                        player.player AS name,
                                        clubs.club,
                                        player.goals,
                                        player.own_goals,
                                        player.shots,
                                        player.xG, 
                                        player.time,
                                        player.position,
                                        player.h_a,
                                        player.yellow_card,
                                        player.red_card,
                                        player.key_passes,
                                        player.assists,
                                        player.xA,
                                        player.xGChain,
                                        player.xGBuildup
                    FROM lineup_stats AS player
                    JOIN clubs ON clubs.club_id = player.team_id AND clubs.season = game.season
                    JOIN general_game_stats AS game ON game.id = player.match_id
                    """,con)
        con.close()
        lineup['date'] = pd.to_datetime(lineup['date'])
        lineup = lineup.dropna(subset=['date'])
        return lineup
    
    def get_transfer(self):
        con = sql.connect('data/transfermarket/transfermarket.db')
        transfer = pd.read_sql("""SELECT    players.name,
                                    players.date_of_birth,
                                    players.sub_position,
                                    players.position,
                                    players.foot,
                                    players.height_in_cm,
                                    player_vals.date,
                                    player_vals.market_value_in_eur AS market_value_in_eur_x
                            FROM players
                            JOIN player_valuations AS player_vals ON player_vals.player_id = players.player_id
                            ORDER BY players.name, player_vals.date
                            """,con)
        con.close()
        transfer['date'] = pd.to_datetime(transfer['date'])
        transfer = transfer.dropna(subset=['date'])
        return transfer
    
    def combine(self):
        # Initialize an empty list to store processed chunks
        chunk_list = []

        chunk_size=100000 #change according to system specs. With this chunk_size, it took ~12 minutes on my computer

        # Process the lineup DataFrame in chunks
        
        for start in tqdm(range(0, len(self.understat_df), chunk_size),leave=True,desc='chunks',colour='green'):
            chunk = self.understat_df.iloc[start:start + chunk_size]
            merged_chunk = self.process_chunk(chunk)
            chunk_list.append(merged_chunk)
            

        # Concatenate all processed chunks
        result = pd.concat(chunk_list).reset_index(drop=True)
        result = result.drop_duplicates()
        save_path = os.getcwd()
        save_path = os.path.join(save_path,'data/main_data')
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)

        save_path = os.path.join(save_path,'main_data.csv')
        result.to_csv(save_path,index=False)


    def process_chunk(self,chunk):
        chunk_sorted = chunk.sort_values(['name', 'date']).reset_index(drop=True)

        merged_list = []
        for name, group in chunk_sorted.groupby('name'):
            if name in self.transfer_df['name'].values:
                transfer_data = self.transfer_df[self.transfer_df['name'] == name]
                transfer_data = transfer_data.sort_values('date').reset_index(drop=True)
                merged_group = pd.merge_asof(
                    group,
                    transfer_data[['date', 'market_value_in_eur_x']],
                    on='date',
                    direction='backward'
                )
                merged_list.append(merged_group)
        
        return pd.concat(merged_list)
