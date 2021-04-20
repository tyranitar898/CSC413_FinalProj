from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import boxscorematchups
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import boxscoretraditionalv2
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import json
import ast
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

TEAMDICT = {"ATL": 0, "BKN": 1, "BOS": 2, "CHA": 3, "CHI": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9, "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 14, "MIA": 15, "MIL": 16, "MIN": 17, "NOP": 18, "NYK": 19, "OKC": 20, "ORL": 21, "PHI": 22, "PHX": 23, "POR": 24, "SAC": 25, "SAS": 26, "TOR": 27, "UTA": 28, "WAS": 29}

def generateData(fileName):
    # features = [{"name": "GP", "index": 3},{"name": "GS", "index": 4},{"name": "MIN", "index": 5},{"name": "FGPCT", "index": 8},{"name": "FG3M", "index": 9},{"name": "REB", "index": 17}, {
    #     "name": "AST", "index": 18}, {"name": "STL", "index": 19}, {"name": "BLK", "index": 20}, {"name": "TOV", "index": 21}, {"name": "PF", "index": 22}, {"name": "PTS", "index": 23}]
    features = [{"name": "AST", "index": 18}, {"name": "STL", "index": 19}, {"name": "BLK", "index": 20}, {"name": "PTS", "index": 23}]
    X = []
    Xlabels = []
    with open(fileName) as f:
        content = f.readlines()
    for line in content:
        playerDict = ast.literal_eval(line)
        name = playerDict['player']['full_name']
        dataarr = player_minute = playerDict["reg_season_career_total"]['data']
        
        if (len(dataarr) != 0 and len(dataarr[0]) != 0):
            # Minutes more than 4000 players   minute index = 5
            player_minute = playerDict["reg_season_career_total"]['data'][0][5]
            player_gamesplayed = playerDict["reg_season_career_total"]['data'][0][3]
            # print(player_minute, player_gamesplayed)
            if None not in playerDict["reg_season_career_total"]['data'][0]:
                if player_minute > 0: #change for minutes 
                    raw_stat_arr_for_player = playerDict["reg_season_career_total"]['data']

                    if len(raw_stat_arr_for_player) != 0:
                        raw_stat_arr_for_player = raw_stat_arr_for_player[0]

                    # print(name, raw_stat_arr_for_player)

                    thisplayerFeaturesArr = []
                    for feature in features:
                        featureValue = raw_stat_arr_for_player[feature['index']
                                                            ] / player_gamesplayed
                        thisplayerFeaturesArr.append(featureValue)

                    # FG_PCT = raw_stat_arr_for_player[8]
                    # FG3_PCT = raw_stat_arr_for_player[11]
                    # thisplayerFeaturesArr.append(FG_PCT)
                    # thisplayerFeaturesArr.append(FG3_PCT)
                    X.append(thisplayerFeaturesArr)
                    Xlabels.append(name)
    return X, Xlabels

def GMM_pred(n, fileNameStr):
    #  activePlayerCareerRegSeasonStats.txt  allPlayerCareerRegSeasonStats.txt
    X, Xlabels = generateData('./data/' + fileNameStr)
    
    gmm = GaussianMixture(n_components=n).fit(X)
    labels = gmm.predict(X)
    pred = []
    

    player_type_dict = {}
    for i in range(len(labels)):
        # pred.append({"class": int(labels[i]),"name": Xlabels[i]})
        
        p_name = Xlabels[i]
        p_class = int(labels[i])
        player_type_dict[p_name] = p_class
    
        
    return player_type_dict


def boxScoreArrPerPlayerDict(fileName):
    XX = []
    YY = []

    boxscoredict = {} #maps player to their box score for an array of games
    with open(fileName) as f:
        content = f.readlines()
    
    curplayer = ""
    for line in content:
        
        if line[0] == "S":
            pass
        elif line[0] == "{":
            playerDict = ast.literal_eval(line)
            full_name = playerDict['full_name']
            boxscoredict[full_name] = []
            curplayer = full_name
        else:
            arr = line[:-3].split(',') #gets rid of and whether they have video
            relevant_stats = (arr[5:]) #MIN,FGM,FGA,FG_PCT,FG3M,FG3A,FG3_PCT,FTM,FTA,FT_PCT,OREB,DREB,REB,AST,STL,BLK,TOV,PF,PTS,PLUS_MINUS
            #switch above to 5 includes Winloss and oppo team but strings. try later if it helps
            boxscoredict[curplayer].append(relevant_stats)
    # print(boxscoredict["Ivica Zubac"])
    return boxscoredict

def getAllGames(fileName, player_type_dict):
    XX = []
    YY = []

    
    with open(fileName) as f:
        content = f.readlines()
    
    line_count = 0

    #gets all the games in this file
    game_ids=[]
    for line in content:
        if line_count > 0 and line_count%2==1:
            line_seperated = line.split(",")
            game_id = line_seperated[4]
            if game_id not in game_ids:
                game_ids.append(game_id)

        line_count += 1
        
    # gets number of minutes each player played given the game
    with open("data/playerGameLog2018-2019.txt") as f:
        content = f.readlines()

    player_usage_per_game = {}
    curplayer = ""

    progress = 0
    for line in content:
        # progress += 1
        # print(progress/23334)
        if line[0] == "{":
            playerDict = ast.literal_eval(line)
            full_name = playerDict['full_name']
            curplayer = full_name
        elif line[0] == "S" :
            pass
        else :
            
            for g_id in game_ids:
                dataline_seperated = line.split(",")
                g_id_infile = dataline_seperated[2] #files game id
                if g_id == g_id_infile:
                    matchupStr = dataline_seperated[5]
                    team1=matchupStr[0:3]
                    team2=matchupStr[-3:]
                    if g_id in player_usage_per_game:
                        teamsdict = player_usage_per_game[g_id]
                        if teamsdict["team1"] == team1:
                            teamsdict["t1players"].append((curplayer,dataline_seperated[7])) #adds name, minutes played in that game
                        elif teamsdict["team2"] == team1:
                            teamsdict["t2players"].append((curplayer,dataline_seperated[7])) #adds name, minutes played in that game
                        else:
                            print("broken")
                        #print(player_usage_per_game[g_id])
                        
                    else:
                        matchupStr = dataline_seperated[5]
                        team1 = (matchupStr[0:3])
                        team2 = (matchupStr[-3:])
                        win = 0
                        winningTeam = team2
                        
                        if dataline_seperated[6] == "W":
                            win = 1
                            winningTeam = team1
                        
                        winingteamOneHotV = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        winngTeamIndex = TEAMDICT[winningTeam]
                        winingteamOneHotV[winngTeamIndex] = 1

                        t1_index = TEAMDICT[team1]
                        t2_index = TEAMDICT[team2]

                       
                        player_usage_per_game[g_id] = {"team1":team1, "t1players": [(curplayer,dataline_seperated[7])], "team2":team2, "t2players":[], "win": win}
                        
    
    #gets X,Y where each data point is X is the TOPX players categories by GMM (top determined by minutes) and Y is the winning team
    for game in player_usage_per_game:
        game_usage = player_usage_per_game[game]
        team1_sorted = sorted(game_usage["t1players"], key=lambda x: x[1])

        TOPX = 5
        team1_top = team1_sorted[:TOPX]
        top_x_player_classes = []

        names = []
        for p in team1_top:
            name = p[0]
            if name in player_type_dict:
                top_x_player_classes.append(player_type_dict[name])
            else:
                print(name)
                top_x_player_classes.append(-1)
            #try count number of players in each class
        
        if len(top_x_player_classes) < TOPX:
            top_x_player_classes.append(-1)
        
        team2_sorted = sorted(game_usage["t2players"], key=lambda x: x[1])
        team2_top = team2_sorted[:TOPX]
       
        for p in team2_top:
            name = p[0]
            if name in player_type_dict:
                top_x_player_classes.append(player_type_dict[name])
            else:
                print(name)
                top_x_player_classes.append(-1)
        
        if len(top_x_player_classes) < (TOPX * 2):
            top_x_player_classes.append(-1)
        
        
        XX.append(top_x_player_classes)
        
        YY.append(game_usage["win"])

    return XX, YY
        
    


def GMM_NN(kcluster):
    player_type_dict = GMM_pred(kcluster, "allPlayerCareerRegSeasonStats.txt")
    # #maps type: [players of that type]

    # boxscoresGivenPlayerDict = boxScoreArrPerPlayerDict("data/playergamelog2018.txt")

    X, Y = getAllGames("data/gamelog_season=2019.txt", player_type_dict)

    XX = np.array(X)
    YY = np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=42)

    clf = MLPClassifier(random_state=1, max_iter=500, batch_size = 100, solver="adam", learning_rate_init=0.005).fit(X_train, y_train)
    score = (clf.score(X_test, y_test))
    return score

for i in range(30, 40):
    cur_score = GMM_NN(i)
    print("iteration {} with {} clusters GMM_NN gets prediction accuracy: {}".format(i-30, i, cur_score))

# cur_score = GMM_NN(25)
# print("using {} clusters GMM_NN gets prediction accuracy: {}".format(25, cur_score))
