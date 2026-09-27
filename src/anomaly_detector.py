import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class RigSensorAnomalyTracker:
    def __init__(self, contamination=0.05, rolling_window=5):
        """
        Initializes the industrial anomaly tracking pipeline.
        :param contamination: Expected proportion of outliers/anomalies in the data.
        :param rolling_window: Window size for calculating moving statistics.
        """
        self.contamination = contamination
        self.rolling_window = rolling_window
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1)
        self.feature_cols = []

    def engineer_sensor_features(self, df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
        """
        Transforms raw time-series sensor vectors into rolling statistical features.
        Mimics real-time sensor processing pipelines used on active rigs.
        """
        processed_df = df.copy()
        engineered_features = []

        for col in sensor_cols:
            # 1. Rolling Mean: Captures shifting baselines / system degradation
            mean_col = f"{col}_rolling_mean"
            processed_df[mean_col] = processed_df[col].rolling(
                window=self.rolling_window, min_periods=1).mean()

            # 2. Rolling Std Dev: Captures sudden spikes / erratic mechanical vibrations
            std_col = f"{col}_rolling_std"
            processed_df[std_col] = processed_df[col].rolling(
                window=self.rolling_window, min_periods=1).std().fillna(0)

            engineered_features.extend([mean_col, std_col])

        self.feature_cols = engineered_features
        return processed_df

    def fit_pipeline(self, train_df: pd.DataFrame, sensor_cols: list):
        """
        Scales engineered features and fits the Isolation Forest model.
        """
        print("[INFO] Processing training telemetry data...")
        fe_df = self.engineer_sensor_features(train_df, sensor_cols)

        # Scale the engineered moving stats matrix
        scaled_features = self.scaler.fit_transform(fe_df[self.feature_cols])

        print("[INFO] Fitting Isolation Forest model to structural sensor baselines...")
        self.model.fit(scaled_features)
        print("[INFO] Model training complete successfully.")

    def monitor_stream(self, stream_df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
        """
        Ingests a fresh telemetry stream, applies transformations, and scores data points.
        Returns data frame with anomaly scores and boolean flags.
        """
        if not self.feature_cols:
            raise ValueError(
                "Pipeline has not been trained. Run fit_pipeline first.")

        fe_df = self.engineer_sensor_features(stream_df, sensor_cols)
        scaled_features = self.scaler.transform(fe_df[self.feature_cols])

        # Isolation Forest outputs: 1 for normal operation, -1 for anomalous variance
        raw_predictions = self.model.predict(scaled_features)
        anomaly_scores = self.model.decision_function(scaled_features)

        # Structure metrics into clear operational fields
        fe_df["anomaly_score"] = anomaly_scores
        # Map output: True if an anomaly is actively flagged (-1)
        fe_df["is_anomaly"] = np.where(raw_predictions == -1, True, False)

        return fe_df


# Example Execution block to verify script logic with simulated data
if __name__ == "__main__":
    print("--- Simulating Offshore Rig Telemetry Matrix ---")

    # Generate mock continuous multi-variable timeline (e.g., Timestamp, Pump Pressure, Shaft Temperature)
    np.random.seed(42)
    timestamps = pd.date_range(
        start="2026-09-27 00:00", periods=100, freq="min")

    mock_data = {
        "timestamp": timestamps,
        "hydraulic_pressure_psi": np.random.normal(loc=3000, scale=50, size=100),
        "shaft_vibration_hz": np.random.normal(loc=60, scale=2, size=100)
    }

    df = pd.DataFrame(mock_data)
    sensors = ["hydraulic_pressure_psi", "shaft_vibration_hz"]

    # Introduce a simulated mechanical shock/anomaly into final indices
    df.loc[95:98, "hydraulic_pressure_psi"] = 3450  # Hazardous pressure spike
    df.loc[96:99, "shaft_vibration_hz"] = 85        # Severe friction shake

    # Initialize and execute pipeline
    tracker = RigSensorAnomalyTracker(contamination=0.05, rolling_window=5)

    # Train pipeline on normal baseline range (indices 0 to 80)
    tracker.fit_pipeline(df.iloc[0:80], sensors)

    # Monitor ongoing streaming window (indices 80 to 100)
    monitored_results = tracker.monitor_stream(df.iloc[80:100], sensors)

    # Filter and view system telemetry alerts
    alerts = monitored_results[monitored_results["is_anomaly"] == True]
    print(
        f"\n[ALERT SUMMARY] Identified {len(alerts)} mechanical anomalies in telemetry window:")
    print(alerts[["timestamp", "hydraulic_pressure_psi",
          "shaft_vibration_hz", "anomaly_score"]])
