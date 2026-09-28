# Real-Time Rig Machinery Predictive Maintenance Pipeline (NASA C-MAPSS) 🌊⚓

## Core Product Showcase
This repository hosts a production-grade, unsupervised machine learning product designed to ingest massive multi-variable sensor streams from heavy rotating industrial components and automatically identify thermodynamic and mechanical degradation cycles.

Instead of relying on mock scripts, this architecture is fully integrated with the **Official NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)**. It tracks 21 distinct continuous subsea/aerodynamic sensors across a fleet of industrial units operating under varying levels of structural duress.

## Machine Learning Architecture
1. **Fleet Partitioned Feature Engineering:** Groups high-variance streaming data cycles by independent machinery IDs to calculate structural `rolling window mean` and `standard deviation matrices` without data leakage.
2. **Unsupervised Outlier Isolation:** Implements an optimized `Isolation Forest` mathematical framework calibrated to flag early structural deviations before equipment breakdown.
3. **Automated Analytics Logging:** Pipelines transformed features directly into structured outputs, generating automated critical alert matrices for control-room operators.

## Performance Metrics Output
The framework scales and checks real telemetry files natively, mapping exact anomalous thresholds, isolating structural failures, and exporting full telemetry audits to an industrial `.csv` schema.
