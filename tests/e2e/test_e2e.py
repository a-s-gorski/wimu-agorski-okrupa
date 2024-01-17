import pytest
import requests
import torch

def test_api_response():
    # Replace 'url' with the actual URL of the API endpoint you want to test
    url = 'http://localhost:8000/predict'
    
    classlist_support = ['Organ', 'Flute', 'Trumpet']

    support_data = {
            'audio': torch.randn(size=[15, 1, 16000]).tolist(),
            'target': [0 for _ in range(5)] + [1 for _ in range(5)] + [2 for _ in range(5)],
            'classlist': classlist_support,
        }
    query_data = {
        'audio': torch.randn(size=[45, 1, 16000]).tolist(),
    }


    response = requests.get(url, json={"support": support_data, "query": query_data}, timeout=300)
    assert response.status_code == 200
    
    response_json = response.json()
    assert 'logits' in response_json and 'predicted_labels' in response_json and 'predicted_classes' in response_json


test_api_response()