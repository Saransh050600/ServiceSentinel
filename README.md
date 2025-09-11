# Service Sentinel AI – Render Deployment

This project hosts a machine learning model that predicts asset failure risk for the Service Sentinel project using Salesforce telemetry data.

## Deployment Steps
1. Train locally with:
   ```bash
   python train_model.py
   ```
2. Push to GitHub and deploy on Render.
3. Use Render's URL in Salesforce Named Credential.
