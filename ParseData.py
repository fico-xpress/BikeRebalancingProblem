# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:08:40 2024

@author: Marco
"""

import os
import sys
import xml
import requests
import itertools
import numpy as np
import pandas as pd
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    nr_outgoing, nr_incoming = parse_trip_data()
    stations = parse_station_location_data()

    net_demand_df = get_net_bike_demands(nr_outgoing.copy(), nr_incoming.copy())

    stations1 = set(list(nr_outgoing[["Start station number", "Start station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations2 = set(list(stations[["terminalName", "name"]].itertuples(index=False, name=None)))
    print(len(stations1))
    print(len(stations2))

    pprint(stations1.difference(stations2))
    pprint(stations2.difference(stations1))


def get_net_bike_demands(nr_outgoing, nr_incoming):

    # Drop "Start" and "End" from dataframes
    nr_outgoing.columns = nr_outgoing.columns.str.replace('^Start ', '', regex=True)
    nr_incoming.columns = nr_incoming.columns.str.replace('^End ', '', regex=True)

    # Merge on keys
    merged_df = pd.merge(nr_outgoing, nr_incoming, on=['date', 'station number', 'station'], suffixes=('_out', '_in'))

    # Calculate the net demand
    merged_df['CLASSIC_net'] = merged_df['CLASSIC_in'] - merged_df['CLASSIC_out']
    merged_df['PBSC_EBIKE_net'] = merged_df['PBSC_EBIKE_in'] - merged_df['PBSC_EBIKE_out']

    # Select relevant columns
    net_demand_df = merged_df[['date', 'station number', 'station', 'CLASSIC_net', 'PBSC_EBIKE_net']]

    # Sanity check:
    print("Net demand:")
    print(net_demand_df)
    print(net_demand_df.groupby("date")['CLASSIC_net'].sum())
    print(net_demand_df.groupby("date")['PBSC_EBIKE_net'].sum())
    return net_demand_df


def parse_station_location_data():
    url = 'https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml'
    response = requests.get(url)
    xml_data = response.content

    root = xml.etree.ElementTree.fromstring(xml_data)

    stations = []
    for station in root.findall('station'):
        station_data = {
            'id': station.find('id').text,
            'name': station.find('name').text,
            'terminalName': station.find('terminalName').text,
            'lat': station.find('lat').text,
            'long': station.find('long').text,
            'installed': station.find('installed').text,
            'locked': station.find('locked').text,
            'installDate': station.find('installDate').text,
            'removalDate': station.find('removalDate').text,
            'temporary': station.find('temporary').text,
            'nbBikes': station.find('nbBikes').text,
            'nbStandardBikes': station.find('nbStandardBikes').text,
            'nbEBikes': station.find('nbEBikes').text,
            'nbEmptyDocks': station.find('nbEmptyDocks').text,
            'nbDocks': station.find('nbDocks').text
        }
        stations.append(station_data)

    print("Nr stations:", len(stations))
    df = pd.DataFrame(stations)
    return df


def parse_trip_data():
    filename = "394JourneyDataExtract15Apr2024-30Apr2024.csv"
    folder = "./"

    # Read data from csv into DataFrame
    data = pd.read_csv(os.path.join(folder, filename), parse_dates=["Start date", "End date"], dtype=str).dropna()

    # Drop times from datetimes to obtain dates
    data["Start datetime"] = data["Start date"]
    data["End datetime"] = data["End date"]
    data["Start date"] = data["Start date"].dt.date
    data["End date"] = data["End date"].dt.date

    # Filter on valid dates
    valid_dates = get_valid_dates(data)
    data = data[data['Start date'].isin(valid_dates)]
    data = data[data['End date'].isin(valid_dates)]

    # Drop rows where trips end on a different day than when they started
    data = data[data['Start date'] == data['End date']]

    # Count types of bikes incoming / outgoing per day per station
    nr_outgoing = data.groupby(["Start date", "Start station number", "Start station"])["Bike model"].value_counts().unstack(fill_value=0)
    nr_incoming = data.groupby(["End date", "End station number", "End station"])["Bike model"].value_counts().unstack(fill_value=0)

    # Ensure each combination of [day x station_id] exists
    nr_outgoing = ensure_all_stations_exist(data, nr_outgoing, "Start")
    nr_incoming = ensure_all_stations_exist(data, nr_incoming, "End")
    assert nr_outgoing.shape == nr_incoming.shape

    nr_outgoing = nr_outgoing.reset_index(drop=True)
    nr_incoming = nr_incoming.reset_index(drop=True)

    return nr_outgoing, nr_incoming


def ensure_all_stations_exist(raw_data, df, point):
    # Get all valid combinations that should exist
    all_combinations = get_all_valid_date_station_combinations(raw_data)

    # Rename "date" to "Start date" (or "End date")
    all_combinations = all_combinations.rename(columns=
                               {"Station number": "%s station number"%point, 
                                "Station": "%s station"%point,
                                "Date": "%s date"%point,
                                })

    # Ensure all valid combinations exist and fill counts with zero
    result = pd.merge(all_combinations, df, 
                      on=['%s date'%point, '%s station number'%point, '%s station'%point], 
                      how='left').fillna(0)
    return result


def get_all_valid_date_station_combinations(raw_data):
    # Get all unique start/end dates and start/end station numbers
    dates = set(raw_data['Start date'].unique()).union(\
            set(raw_data['End date'].unique()))
    station_numbers = set(raw_data['Start station number'].unique()).union(\
                      set(raw_data['End station number'].unique()))

    print("Nr stations:", len(station_numbers))

    # Get all combinations of dates and station_numbers
    all_combinations = pd.DataFrame(list(itertools.product(dates, station_numbers)),
                                columns=['Date', 'Station number'])

    # Obtain station id to station name mapping
    station_nr_to_name_mapping = get_all_stations_info(raw_data)

    # Add station name column to all_combinations
    all_combinations = pd.merge(all_combinations, station_nr_to_name_mapping, on=['Station number'])

    return all_combinations


def get_all_stations_info(raw_data):
    start_station_nr_to_name_mapping = raw_data.set_index('Start station number')['Start station'].to_dict()
    end_station_nr_to_name_mapping = raw_data.set_index('End station number')['End station'].to_dict()
    start_station_nr_to_name_mapping.update(end_station_nr_to_name_mapping)
    station_nr_to_name_mapping = pd.DataFrame(list(start_station_nr_to_name_mapping.items()), columns=['Station number', 'Station'])
    return station_nr_to_name_mapping


def get_valid_dates(df):
    unique_stations = set(df['Start station'].unique())

    grouped = df.groupby('Start date')

    # Filter out groups that do not contain almost all unique stations
    valid_start_dates = []
    for start_date, group in grouped:
        day_unique_stations = set(group['Start station'].unique())
        diff = unique_stations.symmetric_difference(day_unique_stations)
        if len(diff) < 0.05 * len(unique_stations):
            valid_start_dates.append(start_date)

    return valid_start_dates

if __name__ == '__main__':
    main()

