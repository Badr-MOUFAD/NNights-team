"""Script to scrap holidays from Wikipedea."""

import requests
import re

import pandas as pd
from bs4 import BeautifulSoup


# stores
dict_holidays = {
    "date": [],
    "name": []
}

dict_month_day_duration = {
    "month": [],
    "start": [],
    "end": [],
    "duration": [],
    "month-day": []
}

URL = "https://en.wikipedia.org/wiki/Federal_holidays_in_the_United_States"


def main():  # noqa
    # fetch wiki page
    page = requests.get(URL)

    # build beautiful soup obj
    html_page = BeautifulSoup(page.content, "html.parser")

    # select holiday table
    holiday_table = html_page.find("table", class_="wikitable")

    # extract from table

    def style_selector(s):  # noqa
        return s != "background:#efefef;"

    for row in holiday_table.find_all("tr", attrs={"style": style_selector}):
        # extract useful infos
        holiday_date = row.find("th").text
        holiday_name = row.find("td").text

        # save
        dict_holidays["date"].append(holiday_date)
        dict_holidays["name"].append(holiday_name)

    # put dict in df
    df = pd.DataFrame(dict_holidays)

    # remove \n and (.)
    pattern = r'\((.*?)\)|\n'

    df["date"] = df["date"].apply(lambda s: re.sub(pattern, repl="", string=s))
    df["name"] = df["name"].apply(lambda s: re.sub(pattern, repl="", string=s))

    # extract month, start day, duration
    for i in range(len(df)):
        # get entry
        holiday_date = df.loc[i, "date"]

        # split
        pattern = r'\ |–'
        month, *day = re.split(pattern, string=holiday_date)

        # get month num
        month_num = pd.Timestamp(f"{month} 2000").month
        dict_month_day_duration["month"].append(month_num)

        # get start duration day
        start, end = int(day[0]), int(day[-1])

        dict_month_day_duration["start"].append(start)
        dict_month_day_duration["end"].append(end)
        dict_month_day_duration["duration"].append(end-start + 1)

    # arrange in df and concat
    df_extra_info = pd.DataFrame(dict_month_day_duration)
    df_holidays = pd.concat([df, df_extra_info], axis=1)

    # save
    df_holidays.to_csv("./data/usa_holidays.csv", index=False)
    return


# to prevent running script when imported
if '__name__' == "__main__":
    main()
