from utils import FLOORS_TO_NUMS, REG_GROUPS, CAT_FEATURES, preprocess_floor
import pandas as pd
import re
import numpy as np

def preprocess_df(df, use_target=True):
    df = df[df['price_type'] == 1].reset_index(drop=True)
    
    df['date'] = df['date'].apply(lambda x: x.replace('-', '')).astype(int)
    df['id'] = df['id'].apply(lambda x: x.replace('COL_', '')).astype(int)
    
    df.loc[df.osm_city_nearest_population.isna(), ['osm_city_nearest_population', 'osm_city_nearest_name']] = [604901.0, 'Владивосток']
    if use_target:
        target = df['per_square_meter_price'].values
        df = df.drop('per_square_meter_price', axis=1)
    
    df['floor_number'] =  df.floor.apply(lambda x: FLOORS_TO_NUMS.get(x, 1))
    df = preprocess_floor(df)
    df['reg_groups'] = df.region.apply(lambda x: REG_GROUPS[x])
    df = df.drop(['street'], axis=1)
    
    df['city'] = df['city'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))
    df['osm_city_nearest_name'] = df['osm_city_nearest_name'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))
    df['region'] = df['region'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))
    df['reg_groups'] = df['reg_groups'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))
    
    for feat in CAT_FEATURES:
        data_temp = pd.get_dummies(df[feat], drop_first=True)
        df.drop(feat, axis=1, inplace=True)
        data_temp.columns = [feat + '_' + str(col) for col in list(data_temp)]
        df = pd.concat([df, data_temp], axis=1)
    
    feat_imp = pd.read_csv('9th_place_sol_feat_imp.csv')
    feat_names = feat_imp[feat_imp.imp > 0].name.tolist()
    
    df = df[feat_names]
    if use_target:
        target = np.log(target)
    else:
        target = None
    return df, target
    
    
    
    
    
    
    
    