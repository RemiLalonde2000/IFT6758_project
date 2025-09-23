import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures as cf
from os import remove

import requests
import json
import os
from dotenv import load_dotenv

class Data:
    def __init__(self):

        load_dotenv()

        self.data_path = os.getenv("DATA_PATH")
        self.data_path = "../."+self.data_path
        os.makedirs(self.data_path, exist_ok=True)
        if self.data_path is None:
            raise ValueError("Environment variable DATA_PATH is not set.")

        self.session = requests.Session()
        self.max_workers = 16
        self.data = self.get_all_games_id()
        self.big_file_path = 'play_by_play.json'

    def __add__(self, season):

        if not isinstance(season, str):
            raise TypeError("Season must be a string of format YYYY-YYYY")

        self.load_data_local(season)
        return self

    def get_all_games_id(self):
        file_path = os.path.join(self.data_path, 'all_games.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                all_data = json.load(file)
        else:
            response = self.session.get("https://api.nhle.com/stats/rest/en/game")
            if response.status_code == 200:
                all_data = response.json()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=4)
            else:
                raise RuntimeError(response.status_code)
        return all_data


    def get_games_id_from_season(self, season_start):
        id_list_game = []

        for element in self.data['data']:
            if str(element['id'])[:4] in season_start:
                id_list_game.append(element['id'])
        return id_list_game

    def fetch_one_game_pbp(self, game_id):
        try:
            pbp = self.session.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
            if pbp.status_code == 404:
                return None, game_id, 404
            pbp.raise_for_status()
            return pbp.json(), game_id, None
        except requests.exceptions.RequestException as e:
            return None, game_id, e


    def load_data_local(self, season, merge_one_file=True):

        if isinstance(season, str):
            season = [season]
        elif isinstance(season, list):
            season = season
        else:
            raise ValueError("Season must be string or list")

        pbp_big_file = None
        for s in season:

            season_start = s.split("-")[0]
            id_list_games = self.get_games_id_from_season(season_start)
            total_games = len(id_list_games)
            futures = []
            season_dir = os.path.join(self.data_path, s)
            os.makedirs(season_dir, exist_ok=True)

            games_not_fetch = []
            for gid in id_list_games:
                out_path = os.path.join(season_dir, f"{gid}.json")
                if not os.path.exists(out_path):
                    games_not_fetch.append(gid)

            print(f"=== Fetching {len(games_not_fetch)} games from season {s} ===")
            if not games_not_fetch:
                print(f"Season {s} already load in data folder")
                continue

            play_by_play_result = {}
            missing_game = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as e:
                count = 0
                for ids in games_not_fetch:
                    futures.append(e.submit(self.fetch_one_game_pbp, ids))

                for i, completed in enumerate(cf.as_completed(futures)):
                    pbp, game_id, e = completed.result()

                    if e is None and pbp is not None:
                        out_path = os.path.join(season_dir, f'{game_id}.json')
                        Data.save_json(pbp, out_path)

                        if merge_one_file:
                            play_by_play_result[game_id] = pbp
                    else:
                        missing_game[game_id] = e

                    count = count + 1
                    if count%50 == 0 or count == total_games:
                        print(f"{count}/{total_games} games fetched")

            if merge_one_file and play_by_play_result:
                self.add_data_to_big_file(pbp_big_file, s, play_by_play_result)


    def add_data_to_big_file(self, pbp_big_file, s, play_by_play_result):
        if pbp_big_file is None:
            pbp_big_file = self.load_big_pbp_file()
        if s not in pbp_big_file:
            pbp_big_file[s] = {}

        pbp_big_file[s].update(play_by_play_result)
        Data.save_json(pbp_big_file, os.path.join(self.data_path, self.big_file_path))

    def load_big_pbp_file(self):
        path = os.path.join(self.data_path, self.big_file_path)
        os.makedirs(self.data_path, exist_ok=True)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_data(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                file = json.load(file)
            return file
        else:
            raise ValueError("File path doesnt exist")

    @staticmethod
    def save_json(data_file, file_path):
        with open(file_path, "w") as f:
            json.dump(data_file, f)

    def remove_season(self, season):
        season_path = os.path.join(self.data_path, season)
        big_file_path = os.path.join(self.data_path, self.big_file_path)

        if os.path.exists(season_path):
            shutil.rmtree(season_path)


        if os.path.exists(big_file_path):
            with open(big_file_path, "r") as f:
                file = json.load(f)
        else:
            raise ValueError("File not found")

        if season in file:
            del file[season]
            Data.save_json(file, big_file_path)


if __name__=="__main__":
    d = Data(max_workers=16)
    d.load_data_local(['2016-2017', '2017-2018', '2018-2019', '2020-2021', '2021-2022', '2022-2023', '2023-2024',], merge_one_file=True)
    # data = d.get_data('./data/play_by_play.json')

    # d.load_data_local(['2016-2017'], merge_one_file=True)
    # d = d + '2018-2019'
    data = d.get_data(os.path.join(d.data_path, 'play_by_play.json'))
    print(len(data))