# Rig Machinery Sensor Anomaly Tracker 🌊⚓

## Project Overview
This repository contains a production-ready, unsupervised data pipeline designed to ingest streaming telemetry data from offshore rig systems (such as high-pressure hydraulic pumps, subsea drilling strings, or maritime thrusters) and automatically identify early structural failures or erratic operational signatures.

By converting raw time-series data streams into robust rolling statistical baselines, this architecture isolates mechanical noise and triggers diagnostic anomaly alerts before catastrophic equipment downtime occurs.

## Core Engineering Features
- **Rolling Window Feature Engineering:** Automatically computes moving statistical variations (rolling mean and standard deviation matrices) to handle shifting system drift and isolate baseline machinery frequencies.
- **Unsupervised Anomaly Isolation:** Deploys an optimized Scikit-Learn `Isolation Forest` model, making it highly effective for asset environments where labeled operational historical failure metrics are unavailable.
- **Real-Time Streaming Simulation:** Tailored to process multi-variable continuous logs, computing health flags, and generating metric scores natively.

## Technical Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn
- **Developer Workflow:** Git, Bash/PowerShell script automation

## Industrial Transition Framing
*Architectural Insight:* The code architecture deployed within this project applies the exact statistical processing models used to monitor unstable vital parameters within high-stakes environments (such as intensive care multi-variable streaming networks) and maps them directly onto upstream mechanical asset telemetry dashboards.
