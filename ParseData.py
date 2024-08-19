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
from scipy.spatial.distance import pdist, squareform

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 150)

FILENO = "394"

def main():
    nr_stations = None

    stations = parse_station_location_data()
    print(stations.columns)
    distance_matrix = calculate_distance_matrix(stations)
    print(distance_matrix.head())
    sys.exit()
    nr_outgoing, nr_incoming = parse_trip_data(stations, nr_stations)
    parse_trip_data_into_matrix(stations, nr_stations)

    net_demand_df = get_net_bike_demands(nr_outgoing.copy(), nr_incoming.copy())
    dump_trip_df_to_csv(net_demand_df, nr_stations)

    stations = drop_unneeded_station_info(stations, nr_outgoing)
    dump_station_info_to_csv(stations, nr_stations)


    stations1 = set(list(nr_outgoing[["Start station number", "Start station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations2 = set(list(stations[["terminalName", "name"]].itertuples(index=False, name=None)))
    print("Nr stations trips data:", len(stations1))
    print("Nr stations station data:", len(stations2))

    print("\nIn trip but not in stations list:")
    pprint(stations1.difference(stations2))
    print("\nIn stations list but not in trips:")
    pprint(stations2.difference(stations1))
    assert len(stations1.symmetric_difference(stations2)) == 0, "Sets must be identical"



def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def calculate_distance_matrix(stations):
    coords = stations[['lat', 'long']].values.astype(float)
    dist_matrix = squareform(pdist(coords, lambda u, v: haversine(u[0], u[1], v[0], v[1])))
    print(np.min(dist_matrix[dist_matrix>0.0]))
    dist_matrix = dist_matrix / np.min(dist_matrix[dist_matrix>0.0]) / 10
    return pd.DataFrame(dist_matrix, index=stations['name'], columns=stations['name'])

def extract_submatrix(distance_matrix, station_names):
    return distance_matrix.loc[station_names, station_names]


def drop_unneeded_station_info(stations, nr_outgoing):
    stations1 = set(list(nr_outgoing[["Start station number", "Start station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations2 = set(list(stations[["terminalName", "name"]].itertuples(index=False, name=None)))
    
    keys_df = pd.DataFrame(index=stations.index)
    keys_df['Keys'] = list(zip(stations["terminalName"], stations["name"]))
    stations = stations[~keys_df['Keys'].isin(stations2 - stations1)]
    return stations


def dump_station_info_to_csv(stations, nr_stations):
    stations = stations.rename(columns={"terminalName": "station number"})
    stations = stations[['station number', 'nbDocks']]
    stations = stations.sort_values(["station number"], ignore_index=True)
    print("Dumped:", stations.shape)
    
    filename = 'Station_Info.csv'
    if nr_stations is not None:
        filename = 'Station_Info_size%d.csv'%nr_stations
    else:
        filename = 'Station_Info.csv'
    stations.to_csv("./data/" + filename, sep=';', index=False)


def dump_trip_df_to_csv(net_demand_df, nr_stations):
    net_demand_df = net_demand_df.drop(columns=["station"])
    net_demand_df['date'] = net_demand_df['date'].astype(str)
    net_demand_df = net_demand_df.sort_values(["date", "station number"], ignore_index=True)
    print("Dumped:", net_demand_df.shape)

    filename = '%s_Net_Data.csv'%FILENO
    if nr_stations is not None:
        filename = '%s_Net_Data_size%d.csv'%(FILENO, nr_stations)
    else:
        filename = '%s_Net_Data.csv'%FILENO
    net_demand_df.to_csv("./data/" + filename, sep=';', index=False)


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
    assert np.all(net_demand_df.groupby("date")['CLASSIC_net'].sum() == 0)
    assert np.all(net_demand_df.groupby("date")['PBSC_EBIKE_net'].sum() == 0)
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

    # print("Nr stations:", len(stations))
    df = pd.DataFrame(stations)
    df.to_csv("./data/stations_all_info.csv", sep=',', index=False)
    return df


def parse_trip_data_into_matrix(stations, nr_stations):
    filename = "%d_Data.csv"
    folder = "./data"

    # Read data from csv into DataFrame
    data = []
    for i in range(391, 395):
        new_data = pd.read_csv(os.path.join(folder, filename%i), parse_dates=["Start date", "End date"], dtype=str).dropna()
        data.append(new_data)
    data = pd.concat(data)

    print("Initial shape:", data.shape)

    # Drop times from datetimes to obtain dates
    data["Start datetime"] = data["Start date"]
    data["End datetime"] = data["End date"]
    data["Start date"] = data["Start date"].dt.date
    data["End date"] = data["End date"].dt.date

    # Filter on valid dates
    print("Dropping rows with different dates:", end=' ')
    valid_dates = get_valid_dates(data)
    data = data[data['Start date'].isin(valid_dates)]
    data = data[data['End date'].isin(valid_dates)]

    # Drop rows where trips end on a different day than when they started
    data = data[data['Start date'] == data['End date']]
    print(data.shape)

    # Drop rows where trips either start or end at stations we do not know the capacity of
    data = drop_removed_stations_from_trips(data, stations)

    if nr_stations is not None:
        all_stations = get_all_stations_info(data)
        station_names = sorted(list(set(list(all_stations.itertuples(index=False, name=None)))))
        stations_to_drop = station_names[nr_stations:]
        data = drop_all_trips_with_stations(data, stations_to_drop)

    # Filter to include only "CLASSIC" bike model trips
    classic_trips = data[data['Bike model'] == 'CLASSIC']
    # # Only do one date 
    # classic_trips = data[data['Start date'] == data['Start date'].iloc[0]]
    
    # Group by 'Start date', 'Start station number', 'Start station', 'End station number', and 'End station'
    grouped = classic_trips.groupby(['Start date', 'Start station number', 'Start station', 'End station number', 'End station']).size().reset_index(name='count')
    
    # Get unique stations
    stations = list(set(grouped['Start station number']).union(set(grouped['End station number'])))
    stations.sort()
    
    # Create a dictionary to map stations to their indices
    station_to_index = {station: idx for idx, station in enumerate(stations)}
    
    # Initialize the matrix
    n_stations = len(stations)
    matrix = {date: np.zeros((n_stations, n_stations), dtype=int) for date in grouped['Start date'].unique()}
    
    # Populate the matrix
    for _, row in grouped.iterrows():
        date = row['Start date']
        start_idx = station_to_index[row['Start station number']]
        end_idx = station_to_index[row['End station number']]
        matrix[date][start_idx, end_idx] += row['count']
    
    for date, data in matrix.items():
        suffix = "" if nr_stations is None else "size%d_"%nr_stations
        date = date.strftime('%Y_%m_%d')
        pd.DataFrame(data).to_csv("./data/" + '%s_matrix_data_%s%s.csv'%(FILENO,suffix,date), sep=';', index=False)
        # break

    # return matrix


def parse_trip_data(stations, nr_stations=None):
    filename = "%d_Data.csv"
    folder = "./data"

    # Read data from csv into DataFrame
    data = []
    for i in range(391, 395):
        new_data = pd.read_csv(os.path.join(folder, filename%i), parse_dates=["Start date", "End date"], dtype=str).dropna()
        data.append(new_data)
    data = pd.concat(data)

    print("Initial shape:", data.shape)

    # Drop times from datetimes to obtain dates
    data["Start datetime"] = data["Start date"]
    data["End datetime"] = data["End date"]
    data["Start date"] = data["Start date"].dt.date
    data["End date"] = data["End date"].dt.date

    # Filter on valid dates
    print("Dropping rows with different dates:", end=' ')
    valid_dates = get_valid_dates(data)
    data = data[data['Start date'].isin(valid_dates)]
    data = data[data['End date'].isin(valid_dates)]

    # Drop rows where trips end on a different day than when they started
    data = data[data['Start date'] == data['End date']]
    print(data.shape)

    # Drop rows where trips either start or end at stations we do not know the capacity of
    data = drop_removed_stations_from_trips(data, stations)

    if nr_stations is not None:
        if nr_stations == 100 and FILENO == "391":
            data = data[data["Start station"] != 'Disraeli Road, Putney']
            data = data[data["End station"] != 'Disraeli Road, Putney']
        if nr_stations == 100 and FILENO == "394":
            data = data[data["Start station"] != 'Victoria Park Road, Hackney Central']
            data = data[data["End station"] != 'Victoria Park Road, Hackney Central']
        all_stations = get_all_stations_info(data)
        station_names = sorted(list(set(list(all_stations.itertuples(index=False, name=None)))))
        stations_to_drop = station_names[nr_stations:]
        data = drop_all_trips_with_stations(data, stations_to_drop)
        
        all_stations2 = get_all_stations_info(data)
        station_names2 = sorted(list(set(list(all_stations2.itertuples(index=False, name=None)))))
        print(len(station_names2), "stations remaining")
        # print(station_names2.symmetric_difference((set(station_names[:nr_stations]))))

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


def drop_removed_stations_from_trips(data, stations):
    station_nr_to_name_mapping = get_all_stations_info(data)

    stations1 = set(list(station_nr_to_name_mapping[["Station number", "Station"]].drop_duplicates().itertuples(index=False, name=None)))
    stations2 = set(list(stations[["terminalName", "name"]].itertuples(index=False, name=None)))
    
    station_names_to_drop = station_nr_to_name_mapping[~station_nr_to_name_mapping["Station"].isin(stations["name"])]
    station_names_to_drop = stations1 - stations2
    print("Dropping trips involving", len(station_names_to_drop), "stations: ", end='')
    return drop_all_trips_with_stations(data, station_names_to_drop)

def drop_all_trips_with_stations(data, station_names_to_drop):
    def drop_mode(mode):
        keys_df = pd.DataFrame(index=data.index)
        keys_df['Keys'] = list(zip(data["%s station number"%mode], data["%s station"%mode]))
        filtered_data = data[~keys_df['Keys'].isin(station_names_to_drop)]
        return filtered_data

    data = drop_mode("Start")
    data = drop_mode("End")
    print(data.shape)
    return data


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

    # print("Nr stations:", len(station_numbers))

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

