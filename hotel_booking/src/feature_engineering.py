
import pandas as pd 


def feature_engineering(df:pd.DataFrame)-> pd.DataFrame:

    df["has_company"]= df["company"].notna().astype(int)
    df = df.drop(columns = ["company"])


    country_keep = df["country"].value_counts()
    country_keep = country_keep[country_keep > 1000].index
    df["country"] = df["country"].where(df["country"].isin(country_keep),"other")

    #df["room_type_match"] = df["assigned_room_type"] == df["reserved_room_type"] leakage mybe ? 

    season_map = {
    "December": "Winter","January": "Winter","February": "Winter",
    "March": "Spring","April": "Spring","May": "Spring",
    "June": "Summer","July": "Summer","August": "Summer",
    "September": "Autumn","October": "Autumn","November": "Autumn"
    }
    df["arrival_season"] = df["arrival_date_month"].map(season_map)
    df = df.drop(columns = ["arrival_date_month"])


    bins = [0, 7, 14, 21, 31]
    labels = ["early", "mid_early", "mid_late", "late"]

    df["arrival_day_period"] = pd.cut(
        df["arrival_date_day_of_month"],
        bins=bins,
        labels=labels
    )
    df = df.drop(columns = ["arrival_date_day_of_month"])


    df["total_guests"] = df["adults"] + df["children"] + df["babies"]


    agent_keep = df["agent"].value_counts().head(20).index
    df["agent"] = df["agent"].where(
        df["agent"].isin(agent_keep),
        "other"
    )
    
    df["agent"] = df["agent"].astype("string")

    return df