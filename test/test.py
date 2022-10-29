import json


def test_read_result():
    response = client.get('/result/1')
    assert response.status_code == 200
