import numpy as np
import pandas as pd
import json
import os

def read_competitions() -> pd.DataFrame:
    COMPETITION_FILE = "../input/open-data/data/competitions.json"
    return pd.read_json("../input/open-data/data/competitions.json")


def read_matches(compe_id: int, season_id: int) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if str(compe_id) not in os.listdir("../input/open-data/data/matches"):
        raise ValueError("指定されたcompetition_idは存在しません。")
    if f"{season_id}.json" not in os.listdir(f"../input/open-data/data/matches/{compe_id}"):
        raise ValueError("指定されたseason_idは存在しません。")

    MATCH_FILE = f"../input/open-data/data/matches/{compe_id}/{season_id}.json"
    with open(MATCH_FILE, encoding="utf-8") as f:
        match_json = json.load(f)

    match_df = pd.json_normalize(match_json)
    home_mgr_df = pd.json_normalize(match_df.to_dict("records"), record_path="home_team.managers", meta="match_id")
    away_mgr_df = pd.json_normalize(match_df.to_dict("records"), record_path="away_team.managers", meta="match_id")

    match_df = match_df.drop(["home_team.managers", "away_team.managers"], axis=1)

    return match_df, home_mgr_df, away_mgr_df


def read_lineups(match_id: int) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if f"{match_id}.json" not in os.listdir("../input/open-data/data/lineups"):
        raise ValueError("指定されたmatch_idのlineupは存在しません。")

    LINEUP_FILE = f"../input/open-data/data/lineups/{match_id}.json"
    with open(LINEUP_FILE, encoding="utf-8") as f:
        lineup_json = json.load(f)

    lineup_df = pd.json_normalize(lineup_json)
    lineup_df = pd.json_normalize(lineup_df.to_dict("records"), record_path="lineup", meta=["team_id", "team_name"])
    cards_df = pd.json_normalize(lineup_df.to_dict("records"), record_path="cards", meta=["player_id", "player_name"])
    positions_df = pd.json_normalize(lineup_df.to_dict("records"), record_path="positions", meta=["player_id", "player_name"])

    lineup_df = lineup_df.drop(["cards", "positions"], axis=1)

    return lineup_df, cards_df, positions_df


def read_events(match_id: int) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if f"{match_id}.json" not in os.listdir("../input/open-data/data/events"):
        raise ValueError("指定されたmatch_idのeventsは存在しません")

    EVENTS_FILE = f"../input/open-data/data/events/{match_id}.json"
    with open(EVENTS_FILE, encoding="utf-8") as f:
        events_json = json.load(f)

    events_df = pd.json_normalize(events_json)
    tactics_lineup_df = pd.json_normalize(events_df.to_dict("records"), record_path="tactics.lineup", meta="id")
    # related_events_df = events_df[events_df["related_events"].notnull()][["id", "related_events"]]
    related_events_list = []
    for index, row in events_df[events_df["related_events"].notnull()][["id", "related_events"]].iterrows():
        for related_event in row["related_events"]:
            related_events_list.append([row["id"], related_event])
    related_events_df = pd.DataFrame(related_events_list, columns=["id", "related_events"])
    shot_freeze_frame_df = pd.json_normalize(events_df.to_dict("records"), record_path="shot.freeze_frame", meta="id")

    events_df["location_x"] = events_df["location"].apply(lambda points: points[0] if isinstance(points, list) else points)
    events_df["location_y"] = events_df["location"].apply(lambda points: points[1] if isinstance(points, list) else points)
    events_df["pass.end_location_x"] = events_df["pass.end_location"].apply(lambda points: points[0] if isinstance(points, list) else points)
    events_df["pass.end_location_y"] = events_df["pass.end_location"].apply(lambda points: points[1] if isinstance(points, list) else points)
    events_df["carry.end_location_x"] = events_df["carry.end_location"].apply(lambda points: points[0] if isinstance(points, list) else points)
    events_df["carry.end_location_y"] = events_df["carry.end_location"].apply(lambda points: points[1] if isinstance(points, list) else points)
    events_df["shot.end_location_x"] = events_df["shot.end_location"].apply(lambda points: points[0] if isinstance(points, list) else points)
    events_df["shot.end_location_y"] = events_df["shot.end_location"].apply(lambda points: points[1] if isinstance(points, list) else points)
    events_df["shot.end_location_z"] = events_df["shot.end_location"].apply(lambda points: points if not isinstance(points, list) else points[2] if len(points)>2 else np.nan)
    events_df["goalkeeper.end_location_x"] = events_df["goalkeeper.end_location"].apply(lambda points: points[0] if isinstance(points, list) else points)
    events_df["goalkeeper.end_location_y"] = events_df["goalkeeper.end_location"].apply(lambda points: points[1] if isinstance(points, list) else points)
    shot_freeze_frame_df["location_x"] = shot_freeze_frame_df["location"].apply(lambda points: points[0])
    shot_freeze_frame_df["location_t"] = shot_freeze_frame_df["location"].apply(lambda points: points[1])

    events_df = events_df.drop(["tactics.lineup", "related_events", "location", "pass.end_location", "carry.end_location", "shot.end_location", "shot.freeze_frame", "goalkeeper.end_location"], axis=1)
    shot_freeze_frame_df = shot_freeze_frame_df.drop("location", axis=1)

    return events_df, tactics_lineup_df, related_events_df, shot_freeze_frame_df


def read_360(match_id: int) -> [pd.DataFrame, pd.DataFrame]:
    if f"{match_id}.json" not in os.listdir("../input/open-data/data/three-sixty"):
        raise ValueError("指定されたmatch_idのthree-sixtyは存在しません")

    THSIXTY_FILE = f"../input/open-data/data/three-sixty/{match_id}.json"
    with open(THSIXTY_FILE, encoding="utf-8") as f:
        thsixty_json = json.load(f)

    visible_area_df = pd.json_normalize(thsixty_json)
    freeze_frame_df = pd.json_normalize(visible_area_df.to_dict("records"), record_path="freeze_frame", meta="event_uuid")

    visible_area_df["visible_area_x"] = visible_area_df["visible_area"].apply(lambda points:[points[i] for i in range(len(points)) if i%2==0])
    visible_area_df["visible_area_y"] = visible_area_df["visible_area"].apply(lambda points:[points[i] for i in range(len(points)) if i%2!=0])
    freeze_frame_df["location_x"] = freeze_frame_df["location"].apply(lambda points: points[0])
    freeze_frame_df["location_y"] = freeze_frame_df["location"].apply(lambda points: points[1])

    visible_area_df = visible_area_df.drop(["visible_area", "freeze_frame"], axis=1)
    freeze_frame_df = freeze_frame_df.drop("location", axis=1)

    return visible_area_df, freeze_frame_df
