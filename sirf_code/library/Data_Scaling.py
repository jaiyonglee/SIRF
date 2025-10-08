# Load basic libraries
import numpy as np
import pandas as pd
from haversine import haversine


## Divide the percentage of position data in a dataframe
def split_train_test(bus_data, test_ratio, num_seed=42):
    # Permuting bus_data with a fixed seed and then reverting to the original seed
    original_state = np.random.get_state()
    np.random.seed(num_seed)
    shuffled_indices = np.random.permutation(len(bus_data))
    np.random.set_state(original_state)

    # Divide the bus_data into training and test sets
    test_set_size = int(len(bus_data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return bus_data.iloc[train_indices], bus_data.iloc[test_indices]


## Identifies the location data retention rate and adjusts it to the desired retention rate
def control_retention_rate(bus_data, retention_rate, num_seed=42):

    bus_data_len = len(bus_data)
    bus_data_nan = bus_data[bus_data['Latitude_truth'].isna()]
    bus_data_nan_len = len(bus_data_nan)
    bus_data_retension_rate = (bus_data_len - bus_data_nan_len) / bus_data_len
    drop_rate = 1 - retention_rate/bus_data_retension_rate
    
    if drop_rate > 0:
        bus_data_hold, bus_data_deleted = split_train_test(bus_data, drop_rate, num_seed=num_seed)
        bus_data_hold_copy = bus_data_hold.copy()
        bus_data_hold_copy[['Latitude', 'Longitude']] = bus_data_hold_copy[['Latitude_truth', 'Longitude_truth']]
        bus_data_deleted.loc[:, ['Latitude', 'Longitude']] = np.nan
        bus_data_return = pd.concat([bus_data_hold_copy, bus_data_deleted]).sort_index()
        print("Seed data rate:", round((1 - len(bus_data_return[bus_data_return['Latitude'].isna()]) / bus_data_len)*100, 2), "%")
        return bus_data_return, bus_data_return[bus_data_return['Latitude'].isna()]['Node']
    
    else:
        print("Lower than the seed data rate :", round(bus_data_retension_rate*100, 2), "%")
        bus_data[['Latitude', 'Longitude']] = bus_data[['Latitude_truth', 'Longitude_truth']].copy()
        print("Seed data rate (Exceeded):", round((1 - bus_data_nan_len / bus_data_len)*100, 2), "%")
        return bus_data.copy(), bus_data_nan['Node']
    

# Finding distance
def find_distance(pos1, pos2):
    try:
        distance = haversine((pos1[0], pos1[1]), (pos2[0], pos2[1]))
    except ValueError as e:
        print(f"ValueError: {e}. Adjusting coordinates...")
        # 좌표 값을 조정하여 범위 내로 변환
        pos1_adj = (((pos1[0] + 90) % 180 - 90), (pos1[1] + 180) % 360 - 180)
        pos2_adj = (((pos2[0] + 90) % 180 - 90), (pos2[1] + 180) % 360 - 180)
        distance = haversine(pos1_adj, pos2_adj)
        print('Range error arose. Adjusted coordinates and recalculated distance.')

    return distance

# Assign node position and distance to link data
def copy_position_to_link_data(data_node, data_link):
    data_link_copy = data_link.copy()
    data_node_copy = data_node.copy()
    
    data_link_copy['Latitude_1'] = data_link_copy['Node1'].map(data_node_copy.set_index('Node')['Latitude'])
    data_link_copy['Longitude_1'] = data_link_copy['Node1'].map(data_node_copy.set_index('Node')['Longitude'])
    data_link_copy['Latitude_2'] = data_link_copy['Node2'].map(data_node_copy.set_index('Node')['Latitude'])
    data_link_copy['Longitude_2'] = data_link_copy['Node2'].map(data_node_copy.set_index('Node')['Longitude'])
    
    data_link_copy['Distance'] = data_link_copy.apply(lambda x: find_distance((x['Latitude_1'], x['Longitude_1']), (x['Latitude_2'], x['Longitude_2'])), axis=1)
    
    return data_link_copy