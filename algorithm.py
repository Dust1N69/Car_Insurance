import pandas as pd
import numpy as np
import datetime as dt
import geojson as gj
from dateutil import parser
import mpu
from shapely.geometry import Point, Polygon
import winsound as ws
from tqdm import tqdm
import pickle
from pathlib import Path


CONFIG = {
    'max_distance': 0.5,  
    'target_year': 2019,
    'data_path': 'your data path',
    'geojson_path': 'your json file path',
    'polygon_cache': "polygon_cache.pkl",
    'beep_freq': 440,  
    'beep_duration': 2000  
}

def load_or_create_polygon(cache_path):
    cache_file = Path(cache_path)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        polygon = Polygon([(23.412, 120.406), (22.891, 120.344), 
                          (23.102, 120.036), (23.222, 120.649)])
        with open(cache_file, 'wb') as f:
            pickle.dump(polygon, f)
        return polygon

def load_data(file_path):
    try:
        return pd.read_csv(
            file_path,
            usecols=["發生時間", "GPS經度", "GPS緯度"],
            dtype={"GPS經度": float, "GPS緯度": float}
        )
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        raise

def load_geojson(file_path):
    try:
        with open(file_path) as f:
            data = gj.load(f)
            return pd.DataFrame({
                'Date': [feat["properties"]["Date_Str"] for feat in data['features']],
                'Time': [feat["properties"]["Time"] for feat in data['features']],
                'long': [feat["properties"]["Longitude"] for feat in data['features']],
                'lat': [feat["properties"]["Latitude"] for feat in data['features']]
            })
    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading GeoJSON: {str(e)}")
        raise

def process_time_column(df, column_name):
    return pd.to_numeric(
        df[column_name].astype(str).str[:2].where(
            df[column_name].astype(str).str[:2].astype(int) <= 23,
            df[column_name].astype(str).str[:1]
        )
    )

def calculate_nearby_points(data_coords, user_coords, max_dist):
    """Vectorized distance calculation using numpy"""
    data_lat, data_lon = data_coords[:, 0], data_coords[:, 1]
    user_lat, user_lon = user_coords[:, 0], user_coords[:, 1]
    
    distances = mpu.haversine_distance(
        (data_lat[:, np.newaxis], data_lon[:, np.newaxis]),
        (user_lat, user_lon)
    )
    return np.sum(distances < max_dist, axis=1)

def main():
    try:
        data_2020 = load_data(CONFIG['data_path'])
        user_df = load_geojson(CONFIG['geojson_path'])
        
        data_2020['發生時間'] = process_time_column(data_2020, '發生時間')
        
        user_df["Date"] = pd.to_datetime(user_df["Date"], format='%Y-%m-%d')
        user_df = user_df[user_df['Date'].dt.year == CONFIG['target_year']].reset_index(drop=True)
        
        polygon = load_or_create_polygon(CONFIG['polygon_cache'])
        
        user_df['in_polygon'] = np.vectorize(
            lambda lat, lon: polygon.contains(Point(float(lat), float(lon)))
        )(user_df['lat'], user_df['long'])
        user_df = user_df[user_df['in_polygon']].drop(columns=['in_polygon']).reset_index(drop=True)
        
        user_df['Time'] = process_time_column(user_df, 'Time')
        user_df = user_df.sort_values('Time').reset_index(drop=True)
        
        data_coords = data_2020[['GPS緯度', 'GPS經度']].to_numpy()
        unique_times = user_df['Time'].unique()
        
        total_points = 0
        for time in tqdm(unique_times, desc="Processing time slots"):
            time_mask = user_df['Time'] == time
            if not time_mask.any():
                continue
                
            user_coords = user_df[time_mask][['lat', 'long']].astype(float).to_numpy()
            data_mask = data_2020['發生時間'] == time
            if not data_mask.any():
                continue
                
            data_time_coords = data_coords[data_mask]
            points = calculate_nearby_points(data_time_coords, user_coords, CONFIG['max_distance'])
            total_points += points.sum()
        
        print(f"Total nearby points found: {total_points}")
        print(user_df)
        
        ws.Beep(CONFIG['beep_freq'], CONFIG['beep_duration'])
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        ws.Beep(880, 500)  

if __name__ == "__main__":
    main()