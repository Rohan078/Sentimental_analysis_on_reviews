
from prefect import flow, task
import subprocess
import os

@task(retries=1)
def run_training_script():
    print("Starting Training and MLflow Logging...")
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception("Training failed")
    print(result.stdout)
    return "Success"

@flow(name="ML Training Workflow")
def training_flow():
    status = run_training_script()
    print(f"Flow completed with status: {status}")

if __name__ == "__main__":
    
    from prefect.client.schemas.schedules import IntervalSchedule
    from datetime import timedelta
    
    print("Serving flow with 24-hour schedule...")
    training_flow.serve(
        name="daily-training-run",
        schedule=IntervalSchedule(interval=timedelta(days=1))
    )
