import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

class EndToEndPredictiveMaintenanceEngine:
    def __init__(self, rolling_window=5):
        self.rolling_window = rolling_window
        self.scaler = StandardScaler()
        # Using RandomForestRegressor to compute exact remaining useful cycles (RUL)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        
        # NASA C-MAPSS data specifications
        self.index_names = ['unit_number', 'time_in_cycles']
        self.setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        self.sensor_names = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names
        
        self.active_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
                               'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_20', 'sensor_21']
        self.feature_cols = []

    def load_and_prepare(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """Loads NASA layout and calculates remaining operational lifetimes."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required data file at: {file_path}")
        
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.col_names)
        
        if is_train:
            # For training data, RUL is calculated by finding the maximum cycle minus current cycle
            max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
            df['RUL'] = max_cycles - df['time_in_cycles']
        return df

    def engineer_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes statistical trend metrics across unit partitions."""
        processed_df = df.copy()
        engineered_features = []

        for sensor in self.active_sensors:
            mean_col = f"{sensor}_roll_mean"
            std_col = f"{sensor}_roll_std"
            
            processed_df[mean_col] = processed_df.groupby('unit_number')[sensor].transform(
                lambda x: x.rolling(window=self.rolling_window, min_periods=1).mean()
            )
            processed_df[std_col] = processed_df.groupby('unit_number')[sensor].transform(
                lambda x: x.rolling(window=self.rolling_window, min_periods=1).std().fillna(0)
            )
            engineered_features.extend([mean_col, std_col])

        self.feature_cols = engineered_features
        return processed_df

    def build_system_pipeline(self, train_path: str, test_path: str, rul_truth_path: str, export_path: str):
        print(f"[1/4 INGESTION] Processing historical training fleet: {train_path}")
        train_df = self.load_and_prepare(train_path, is_train=True)
        train_fe = self.engineer_rolling_features(train_df)
        
        # Scale and fit regression model
        X_train = self.scaler.fit_transform(train_fe[self.feature_cols])
        y_train = train_fe['RUL']
        
        print(f"[2/4 MODELING] Training RUL Regression Engine on {X_train.shape} system data points...")
        self.model.fit(X_train, y_train)
        
        print(f"[3/4 TESTING] Processing active/running live fleet telemetry: {test_path}")
        test_df = self.load_and_prepare(test_path, is_train=False)
        test_fe = self.engineer_rolling_features(test_df)
        
        # For the test set, we evaluate the system state at the *last recorded cycle* of each engine unit
        live_fleet_snapshot = test_fe.groupby('unit_number').last().reset_index()
        X_test = self.scaler.transform(live_fleet_snapshot[self.feature_cols])
        
        # Predict remaining lifetimes
        predicted_rul = self.model.predict(X_test)
        
        # 4/4 VALIDATION: Ingest ground-truth labels directly as a flat array matrix
        print(
            f"[4/4 VALIDATION] Loading ground-truth lifetime baseline from: {rul_truth_path}")
        actual_rul = pd.read_csv(rul_truth_path, header=None)[0].values

        
        # Calculate industrial analytics scores
        mae = mean_absolute_error(actual_rul, predicted_rul)
        r2 = r2_score(actual_rul, predicted_rul)
        
        print("\n=================== END-TO-END PRODUCT METRICS ===================")
        print(f" -> Fleet Telemetry Operational Baseline: Active & Locked.")
        print(f" -> Mean Absolute Error (MAE): {mae:.2f} Operational Cycles")
        print(f" -> Variance Fit Score (R² Score): {r2:.4f}")
        
        # Build evaluation metrics table
        summary_results = pd.DataFrame({
            'unit_number': live_fleet_snapshot['unit_number'],
            'last_recorded_cycle': live_fleet_snapshot['time_in_cycles'],
            'predicted_remaining_cycles': np.round(predicted_rul).astype(int),
            'actual_remaining_cycles': actual_rul
        })
        
        print("\n[LIVE MONITORING CONTROL ROOM] Predicted vs Actual Fleet Lifetimes:")
        print(summary_results.head(5).to_string(index=False))
        
        summary_results.to_csv(export_path, index=False)
        print(f"\n[EXPORT] Full fleet degradation matrix safely saved to: {export_path}")
        print("==================================================================")

if __name__ == "__main__":
    BASE_DIR = os.path.join("data", "raw")
    TRAIN_FILE = os.path.join(BASE_DIR, "train_FD001.txt")
    TEST_FILE = os.path.join(BASE_DIR, "test_FD001.txt")
    RUL_FILE = os.path.join(BASE_DIR, "RUL_FD001.txt")
    OUTPUT_FILE = os.path.join("data", "fleet_rul_predictions_output.csv")
    
    engine = EndToEndPredictiveMaintenanceEngine(rolling_window=5)
    
    try:
        engine.build_system_pipeline(TRAIN_FILE, TEST_FILE, RUL_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"\n [CRITICAL SYSTEM ERROR] {e}")
