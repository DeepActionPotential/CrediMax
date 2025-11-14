# src/pipelines/deployments/train_deployment.py

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from src.pipelines.train_flow import training_flow


# Daily at 2 AM
schedule = CronSchedule(cron="0 15 * * *", timezone="Africa/Cairo")

deployment = Deployment.build_from_flow(
    flow=training_flow,
    name="daily-credit-risk-training",
    schedule=schedule,
    parameters={"config_path": "configs/config.yaml"},
)

if __name__ == "__main__":
    deployment.apply()
