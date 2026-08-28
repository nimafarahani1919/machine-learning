import pandas as pd 
import numpy as np

def data_cleaning(df:pd.DataFrame) -> pd.DataFrame:

    leaks = ["reservation_status", "reservation_status_date","assigned_room_type"]
    df.drop(columns = leaks,inplace=True)

    #df=df.drop_duplicates(keep="first") it has too much duplicateds 

    drops = ["arrival_date_week_number","previous_bookings_not_canceled"]
    df.drop(columns = drops,inplace=True)

    non_negative_columns = [
    "lead_time","stays_in_weekend_nights","stays_in_week_nights","children",
    "babies","adr","required_car_parking_spaces","total_of_special_requests",
    "days_in_waiting_list","previous_cancellations","booking_changes"
    ]
    for column in non_negative_columns:
        df[column] = df[column].where(df[column] >= 0, np.nan)

    row_drop = df.index[(df["adults"]==0) & (df["children"]==0)]
    df.drop(row_drop,inplace=True,axis=0)

    return df