from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import pickle
import utils
import argparse
import yaml
import sys

SAT_COVARS = ['band_25', 'band_26', 'band_27', 'band_28', 'band_29', 'band_30', 'band_31', 'band_33', 'band_34']
TOPO_COVARS = ['aspct_4_band_1', 'elevation_full_10m_3338_band_1', 'maxc_4_band_1', 'sl_4_band_1', 'spi_band_1', 'swi_10_band_1', 'tpi_4_band_1']
CLIMATE_COVARS = ['ppt_annual_band_1', 'tmean_swi_band_1', 'tmin_january_band_1']

def load_data(args):
    # Load K-Folds
    fold_pt = args.fold_pt
    with open(fold_pt, 'r') as file:
        fold_indices = json.load(file)
        
    # Load covars and gt
    point_data_gdf = gpd.read_file(args.sat_data_pt)
    point_data_gdf = point_data_gdf[SAT_COVARS + ['id']]
    topo_covar_gdf = pd.read_csv(args.topo_data_pt)
    topo_covar_gdf = topo_covar_gdf[TOPO_COVARS + ['id']]
    climate_covar_gdf = pd.read_csv(args.climate_data_pt)
    climate_covar_gdf = climate_covar_gdf[CLIMATE_COVARS + ['id']]

    gt_df = pd.read_json(args.json_gt)

    # Preprocessing covars and gt
    gt_df['aksdb_dts'] = pd.to_datetime(gt_df['aksdb_dts'])
    point_data_gdf = gt_df.merge(
        point_data_gdf,
        how="left",
        on="id"
    )

    # Merging clim and topo to main sat + data check
    initial_rows = point_data_gdf.shape[0]
    merged_df = point_data_gdf.merge(topo_covar_gdf, on="id", how="inner")
    assert merged_df.shape[0] == initial_rows, "Row count changed after topo merge"
    merged_df = merged_df.merge(climate_covar_gdf, on="id", how="inner")
    assert merged_df.shape[0] == initial_rows, "Row count changed after climate merge"

    return merged_df, fold_indices
    
def main(args):
    prep_data_df, fold_indices = load_data(args)
    os.makedirs(args.outroot, exist_ok=True)
    
    if args.task_name == "tax_order":
        prep_data_df["tax_order"] = prep_data_df["tax_order"].apply(utils._collapse_value)
        prep_data_df['tax_order'] = prep_data_df["tax_order"].apply(utils._get_tax_order_category)
        prep_data_df = prep_data_df[prep_data_df["tax_order"].notna()]
        y_var = ['tax_order']
        run_func = utils.run_rf_multiclass
        eval_func = utils.run_multi_metric
    elif args.task_name == "nsp":
        prep_data_df["aksdb_pf1m_bin"] = \
            prep_data_df["aksdb_pf1m_bin"].apply(utils._get_aksdb_pf1m_bin)
        prep_data_df = prep_data_df[prep_data_df["aksdb_pf1m_bin"].notna()]
        y_var = ['aksdb_pf1m_bin']
        run_func = utils.run_rf_binary
        eval_func = utils.run_binary_metric
    elif args.task_name == "peat_level":
        prep_data_df = prep_data_df[prep_data_df["peat_level"].notna()]
        prep_data_df["binary_peat"] = prep_data_df["peat_level"].apply(utils._get_peat_level_binary)
        prep_data_df = prep_data_df[prep_data_df["binary_peat"].notna()]
        y_var = ['binary_peat']
        run_func = utils.run_rf_binary
        eval_func = utils.run_binary_metric
    print('Post prep shape: ', prep_data_df.shape)
        
    covars = SAT_COVARS + TOPO_COVARS + CLIMATE_COVARS
    X = prep_data_df[covars].to_numpy()
    y = prep_data_df[y_var].to_numpy()
    indices = prep_data_df['id'].to_list()

    print('X shape: ', X.shape)
    print('Y shape: ', y.shape)
    
    rf11_hyperparams = {
        'est': 253,
        'min_samples_split': 5,
        'max_depth': 16,
        'n_jobs': 4,
        'verbose': 1
    }
    if args.save_weights:
        weights_out = os.path.join(args.outroot, 'weights')
        os.makedirs(weights_out, exist_ok=True)
    else:
        weights_out = None
    run_func(indices, fold_indices, X, y, rf11_hyperparams, \
             outroot = args.outroot, save_rf = weights_out)
    
    #Run both Theresa's original eval and Yijun's eval (has overlap)
    eval_func(indices, fold_indices, y, args.outroot)
    
def parse_args():
    parser = argparse.ArgumentParser(description="RF pipeline with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    args = parser.parse_args()

    # Load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
