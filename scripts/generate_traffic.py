import os
import random
import time

import requests

URL = os.environ.get("API_URL", "http://localhost/predict")
COUNT = int(os.environ.get("COUNT", "300"))

for i in range(COUNT):
    patient = {
        "age": random.randint(35, 77),
        "sex": random.randint(0, 1),
        "cp": random.randint(1, 4),
        "trestbps": random.randint(100, 180),
        "chol": random.randint(150, 400),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(90, 190),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 4), 1),
        "slope": random.randint(1, 3),
        "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }
    response = requests.post(URL, json=patient, timeout=5)
    print(i + 1, response.status_code, response.json())
    time.sleep(0.2)