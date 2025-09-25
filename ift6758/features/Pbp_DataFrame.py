
import os
import sys
sys.path.append(os.path.abspath("."))
from ift6758.GamesUtils import GamesUtils
from ift6758.data.Data import Data
import pandas as pd

class Pbp_to_DataFrame:

    def __init__(self):

        self.data_path = './games_data'

        
    def get_game(self, id):
        season = id[:4] + '-' + str(int(id[:4])+1)
        game = Data.get_data(os.path.join(self.data_path, season, id+'.json'))
        return game
    

    def get_net_and_situation(self, teams, details, event):
        team_event_type = teams.get(details['eventOwnerTeamId'])[1]
        situationCode = event['situationCode']
        away_goalie = int(situationCode[0])
        away_player = int(situationCode[1])
        home_player = int(situationCode[2])
        home_goalie = int(situationCode[3])

        if team_event_type == 'home':
            if away_goalie == 1:
                filet = 'Not empty'
            elif away_goalie == 0:
                filet = 'Empty'
            else:
                filet = None

            if (away_goalie + away_player) == (home_goalie + home_player):
                situation = 'Even strength'
            elif (away_goalie + away_player) > (home_goalie + home_player):
                situation = 'Short-handed'
            elif (away_goalie + away_player) < (home_goalie + home_player):
                situation = 'Power play'
            else:
                situation = None
            

        elif team_event_type == 'away':
            if home_goalie == 1:
                filet = 'Not empty'
            elif home_goalie == 0:
                filet = 'Empty'
            else:
                filet = None

            if (away_goalie + away_player) == (home_goalie + home_player):
                situation = 'Even strength'
            elif (away_goalie + away_player) > (home_goalie + home_player):
                situation = 'Power play'
            elif (away_goalie + away_player) < (home_goalie + home_player):
                situation = 'Short-handed'
            else:
                situation = None
        return filet, situation

    def get_event_type_and_player(self, roaster, details, event):

        if event['typeDescKey'] == 'shot-on-goal':
            player = roaster.get(details['shootingPlayerId'])[0]
            event_type = 'Shot'
        elif event['typeDescKey'] == 'goal':
            player = roaster.get(details['scoringPlayerId'])[0]
            event_type = 'Goal'
        else:
            player = None
            event_type = None

        return player, event_type
    
    def build_game_DataFrame(self, game_id):
        #away goalie (1=in net, 0=pulled)-away skaters-home skaters-home goalie (1=in net, 0=pulled)
        #https://gitlab.com/dword4/nhlapi/-/issues/112?utm_source=chatgpt.com
        game = self.get_game(game_id)
        teams = GamesUtils.get_teams(game)
        roaster = GamesUtils.get_game_roaster(game)

        pbp_df = []
        for event in game['plays']:
            if event['typeDescKey'] == 'shot-on-goal' or event['typeDescKey'] == 'goal':
                details = event['details']
            
                event_id = event['eventId']
                event_team = teams.get(details['eventOwnerTeamId'])[0].split('-')[0]
                goalie = roaster.get(details['goalieInNetId'])[0]
                period = event['periodDescriptor']['number']
                period_time = event['timeInPeriod']
                x = details['xCoord']
                y = details['yCoord']
                shotType = details['shotType']

                player, event_type = self.get_event_type_and_player(roaster, details, event)
                filet, situation = self.get_net_and_situation(teams, details, event)

                row = {'ID': event_id, 
                       'Team':event_team, 
                       'Event Type':event_type, 
                       'Type of Shot':shotType, 
                       'Shooter':player, 
                       'Goalie':goalie,
                       'Net':filet,
                       'Situation':situation,
                       'Period':period,
                       'Time':period_time,
                       'X':x,
                       'Y':y
                    }
                pbp_df.append(row)
        return pd.DataFrame(pbp_df)


if __name__=="__main__":
    c = Pbp_to_DataFrame()
    df = c.build_game_DataFrame('2016010004')
    print(df)
    shots = df[df['Event Type'] == 'Shot']
    print(shots)
    pt = shots.pivot_table(index='Team',
                       columns='Type of Shot',
                       values='ID',
                       aggfunc='count',
                       fill_value=0)
    print(pt)
