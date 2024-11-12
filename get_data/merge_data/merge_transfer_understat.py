import pandas as pd
import os
from tqdm.auto import tqdm
import sqlite3 as sql
import unicodedata

class merge:
    def __init__(self):
        self.lineup_df = self.get_understat()
        self.transfer_df = self.get_transfer()
        self.combined = self.combine()
        self.save_file()
        print('Done merging Transfermarkt and Understat')

    def get_understat(self):
        con = sql.connect('data/understat/understat_game_data/understat_lineup_game_stats.db')

        lineup = pd.read_sql("""SELECT  game.date,
                            LOWER(TRIM(p.player)) as name,
                            LOWER(SUBSTR(p.player,1,INSTR(p.player,' ')-1)) AS first_name,
                            LOWER(SUBSTR(SUBSTR(p.player,INSTR(p.player,' ')+1 ),INSTR(SUBSTR(p.player,INSTR(p.player,' ')+1 ),' ')+1 )) AS last_name,
                            clubs.club,
                            game.season,
                            game.league,
                            p.goals,
                            p.own_goals,
                            p.shots,
                            p.xG, 
                            p.time,
                            p.position AS game_position,
                            p.h_a,
                            p.yellow_card,
                            p.red_card,
                            p.key_passes,
                            p.assists,
                            p.xA,
                            p.xGChain,
                            p.xGBuildup
        FROM lineup_stats AS p
        JOIN clubs ON clubs.club_id = p.team_id AND clubs.season = game.season
        JOIN general_game_stats AS game ON game.id = p.match_id
                    """,con)
        con.close()
        lineup['date'] = pd.to_datetime(lineup['date'])
        lineup['name'] = lineup['name'].str.lower()
        lineup['first_name'] = lineup['first_name'].str.lower()
        lineup['last_name'] = lineup['last_name'].str.lower()
        lineup['name'] = lineup['name'].apply(self.normalize_name)
        lineup['first_name'] = lineup['first_name'].apply(self.normalize_name)
        lineup['last_name'] = lineup['last_name'].apply(self.normalize_name)

        lineup = lineup.dropna(subset=['date'])
        lineup = lineup.sort_values(by='date')
        return lineup
    
    def get_transfer(self):
        con = sql.connect('data/transfermarket/transfermarket.db')
        transfer = pd.read_sql("""WITH clubhistory AS (
                        
                        WITH transfer_history AS (
                            -- Get the transfer history with start and end dates
                        SELECT 
                            t.player_id,
                            COALESCE(LAG(t.transfer_date) OVER (PARTITION BY t.player_id ORDER BY t.transfer_date), DATE(p.date_of_birth)) AS start_date,
                            t.transfer_date AS end_date,
                            c1.name AS club_name
                        FROM 
                            transfers AS t
                        JOIN 
                            clubs AS c1 ON c1.club_id = t.from_club_id
                        JOIN 
                            players AS p ON p.player_id = t.player_id
                    ),
                    current_club AS (
                        -- Get the current club (most recent transfer) with CURRENT_DATE as end date
                        SELECT 
                            t.player_id,
                            t.transfer_date AS start_date,
                            CURRENT_DATE AS end_date,
                            c2.name AS club_name
                        FROM transfers AS t
                        JOIN clubs AS c2 ON c2.club_id = t.to_club_id
                        ORDER BY t.transfer_date DESC
                        LIMIT 1
                    )

                        -- Combine the history and current club
                        SELECT * FROM transfer_history
                        UNION ALL
                        SELECT * FROM current_club
                        ORDER BY start_date, end_date)

                        
                        
                        
                        SELECT  LOWER(TRIM(p.name)) as name,
                                LOWER(SUBSTR(p.name,1,INSTR(p.name,' ')-1)) AS first_name,
                                LOWER(SUBSTR(SUBSTR(p.name,INSTR(p.name,' ')+1 ),INSTR(SUBSTR(p.name,INSTR(p.name,' ')+1 ),' ')+1 )) AS last_name,
                                p.date_of_birth AS date_of_birth,
                                p.sub_position AS sub_position,
                                p.position AS position,
                                p.foot AS foot,
                                p.height_in_cm AS height_in_cm,
                                pv.date AS date,
                                'no' AS transfered,
                                CASE 
                                        WHEN  cb.start_date < pv.date AND pv.date <= cb.end_date THEN cb.club_name
                                        ELSE c.name
                                END AS transfered_from,
                                CASE 
                                        WHEN cb.start_date < pv.date AND pv.date <= cb.end_date THEN cb.club_name
                                        ELSE c.name
                                END AS transfered_to,
                                0.0 AS transfer_fee,
                                pv.market_value_in_eur AS market_value_in_eur
                    FROM players AS p
                    JOIN player_valuations AS pv ON pv.player_id = p.player_id
                    LEFT JOIN clubhistory AS cb ON cb.player_id = p.player_id 
                        AND (cb.start_date < pv.date AND pv.date <= cb.end_date)
                    JOIN clubs AS c ON c.club_id = pv.current_club_id
                    UNION ALL
                    SELECT
                                LOWER(TRIM(p.name)) as name,
                                LOWER(SUBSTR(p.name,1,INSTR(p.name,' ')-1)) AS first_name,
                                LOWER(SUBSTR(SUBSTR(p.name,INSTR(p.name,' ')+1 ),INSTR(SUBSTR(p.name,INSTR(p.name,' ')+1 ),' ')+1 )) AS last_name,
                                p.date_of_birth AS date_of_birth,
                                p.sub_position AS sub_position,
                                p.position AS position,
                                p.foot AS foot,
                                p.height_in_cm AS height_in_cm,
                                t.transfer_date AS date,
                                'yes' AS transfered,
                                c1.name AS transfered_from,
                                c2.name AS transfered_to,
                                t.transfer_fee AS transfer_fee,
                                t.market_value_in_eur AS market_value_in_eur
                    FROM players AS p
                    JOIN transfers AS t ON t.player_id = p.player_id
                    JOIN clubs AS c1 ON c1.club_id = t.from_club_id
                    JOIN clubs AS c2 ON c2.club_id = t.to_club_id
                    ORDER BY name, date
                            """,con)
        con.close()
        transfer['date'] = pd.to_datetime(transfer['date'])
        transfer['name'] = transfer['name'].str.lower()
        transfer['first_name'] = transfer['first_name'].str.lower()
        transfer['last_name'] = transfer['last_name'].str.lower()
        transfer['name'] = transfer['name'].apply(self.normalize_name)
        transfer['first_name'] = transfer['first_name'].apply(self.normalize_name)
        transfer['last_name'] = transfer['last_name'].apply(self.normalize_name)
        transfer = transfer.dropna(subset=['date'])
        transfer = transfer.sort_values(by='date')
        return transfer
    
    def combine(self):
        combined = pd.merge_asof(
                    self.lineup_df,                 
                    self.transfer_df,               
                    by=['last_name','first_name'],              
                    on='date',              
                    direction='nearest',   
                    )
        
        combined = combined.dropna()
        combined = combined.drop(columns=['name_x'])
        combined = combined.rename(columns={'name_y':'name'})
        return combined
    
    def save_file(self):
        save_path = os.getcwd()
        save_path = os.path.join(save_path,'data/main_data')
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path = os.path.join(save_path,'main_data_understat.csv')
        self.combined.to_csv(save_path,index=False)

    def normalize_name(self,name):
        normalized_name = unicodedata.normalize('NFKD',name).encode('ASCII','ignore').decode('utf-8')
        normalized_name = normalized_name.replace('ø','o').replace('Ø','O')
        return normalized_name

