import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IndustrialPredictiveMaintenancePipeline:
    def __init__(self, contamination=0.03, rolling_window=5):
        self.contamination = contamination
        self.rolling_window = rolling_window
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1)

        # Mapping out the official NASA C-MAPSS dataset structural columns
        self.index_names = ['unit_number', 'time_in_cycles']
        self.setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        self.sensor_names = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names

        # Select high-variance sensors for feature engineering to avoid constant noise columns
        self.active_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                               'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_20', 'sensor_21']
        self.feature_cols = []

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Ingests the raw NASA space-separated text layout."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing real dataset asset at: {file_path}. Please download train_FD001.txt.")

        print(
            f"[INGESTION] Loading real-world telemetry asset from {file_path}...")
        df = pd.read_csv(file_path, sep=r'\s+',
                         header=None, names=self.col_names)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms continuous industrial data streams into rolling window metrics."""
        processed_df = df.copy()
        engineered_features = []

        # Group by engine unit number to ensure moving windows do not cross-contaminate separate systems
        for sensor in self.active_sensors:
            mean_col = f"{sensor}_roll_mean"
            std_col = f"{sensor}_roll_std"

            processed_df[mean_col] = processed_df.groupby('unit_number')[sensor].transform(
                lambda x: x.rolling(
                    window=self.rolling_window, min_periods=1).mean()
            )
            processed_df[std_col] = processed_df.groupby('unit_number')[sensor].transform(
                lambda x: x.rolling(window=self.rolling_window,
                                    min_periods=1).std().fillna(0)
            )
            engineered_features.extend([mean_col, std_col])

        self.feature_cols = engineered_features
        return processed_df

    def train_pipeline(self, raw_file_path: str):
        """Fits the data pipeline on the engineered real-world asset baseline."""
        df = self.load_raw_data(raw_file_path)
        fe_df = self.engineer_features(df)

        print(
            f"[TRANSFORMATION] Generated {len(self.feature_cols)} rolling window statistical features.")
        scaled_features = self.scaler.fit_transform(fe_df[self.feature_cols])

        print(
            f"[MODELING] Training Isolation Forest core on {scaled_features.shape[0]} matrix nodes...")
        self.model.fit(scaled_features)
        print("[MODELING] Unsupervised anomaly baseline lock complete.")

    def run_live_inference_simulation(self, raw_file_path: str, output_csv_path: str):
        """Simulates an edge-computing monitoring stream on real-world industrial units."""
        df = self.load_raw_data(raw_file_path)
        fe_df = self.engineer_features(df)

        scaled_features = self.scaler.transform(fe_df[self.feature_cols])
        predictions = self.model.predict(scaled_features)
        scores = self.model.decision_function(scaled_features)

        fe_df['anomaly_score'] = scores
        fe_df['is_anomaly'] = np.where(predictions == -1, True, False)

        # Evaluation Summary Metrics
        total_records = len(fe_df)
        flagged_anomalies = fe_df['is_anomaly'].sum()
        anomaly_rate = (flagged_anomalies / total_records) * 100

        print("\n=================== SYSTEM METRICS EVALUATION ===================")
        print(
            f" -> Total Operational Telemetry Cycles Checked: {total_records}")
        print(
            f" -> Confirmed Anomaly/Degradation Flags Logged: {flagged_anomalies}")
        print(f" -> Asset Disruption Baseline Rate: {anomaly_rate:.2f}%")

        # Filter down to look at critical units nearing systemic failure
        critical_alerts = fe_df[fe_df['is_anomaly'] == True].tail(5)
        print("\n[CRITICAL EDGE ALERTS] Most Recent Systemic Deviations:")
        print(critical_alerts[['unit_number',
              'time_in_cycles', 'anomaly_score', 'is_anomaly']])

        # Save output product array
        fe_df.to_csv(output_csv_path, index=False)
        print(
            f"\n[EXPORT] Processed production logs successfully saved to: {output_csv_path}")
        print("================================================================")


if __name__ == "__main__":
    # Standard repository production directories
    DATA_PATH = os.path.join("data", "raw", "train_FD001.txt")
    OUTPUT_PATH = os.path.join("data", "processed_telemetry_output.csv")

    pipeline = IndustrialPredictiveMaintenancePipeline(
        contamination=0.03, rolling_window=5)

    try:
        pipeline.train_pipeline(DATA_PATH)
        pipeline.run_live_inference_simulation(DATA_PATH, OUTPUT_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ [ERROR] {e}")
        print("👉 Please download the NASA C-MAPSS dataset and place 'train_FD001.txt' in data/raw/\n")
