# ML-Development
Repository for the Machine Learning (ML) Service.

## Overview
This service is developed using **FastAPI** and written in **Python**. It provides a single endpoint, `/predict`, to recommend courses based on a user's skill set.

## Install
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## How to Run
Start the service locally by running:
```bash
python main.py
```

## Endpoint

### `/predict`

- **Method**: POST  
- **Description**: Predicts and recommends course IDs based on the provided user ID and skill set.

#### Request
- **Content-Type**: `application/json`
- **Body**:  
  ```json
  {
      "user_id": 123,
      "skillset": ["skillset_1", "skillset_2", "skillset_3"]
  }

#### Response
- **Content-Type**: `application/json`
- **Body**:  
  ```json
  [101, 102, 103, 104, 105]
