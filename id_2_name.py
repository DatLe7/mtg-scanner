import requests
import json
import time

def get_name(id: str, header: dict):
    '''returns card name if successful else None'''
    time.sleep(0.1)
    url = 'https://api.scryfall.com/cards/' + id
    response = requests.get(url, headers=header)
    if response.ok:
        data = response.json()
        return data.get('name', None)
    else:
        return None

def create(input: str, output: str):
    header={'User-Agent' : 'MTG_Card_Scanner/1.0', 'accept' : 'application/json'}
    id_2_name = {}

    with open(input, 'r') as f:
        idx_2_id = json.load(f)

    for id in idx_2_id.values():
        name = get_name(id, header)
        while name is None:
            name = get_name()
        print(name)
        id_2_name[id] = name

    with open(output, 'w') as output_json:
        json.dump(id_2_name, output_json, indent=4)


create('idx_2_label.json', 'id_2_name.json')

