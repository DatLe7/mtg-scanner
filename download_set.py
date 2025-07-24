import csv
import requests
import time

def download_image(id: str, path: str, header: dict):
    time.sleep(0.1)
    url = f'https://api.scryfall.com/cards/{id}'
    param = {'format' : 'image', 'version' : 'large'}
    response = requests.get(url, params=param, headers=header)
    if response.ok:
        with open(path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False


def build_csv(card_ids: list, img_path: str, csv_name: str, header: dict):
    fields = ['id', 'path']
    csv_data = []

    for idx, id in enumerate(card_ids):
        card_path = img_path + f'card{idx}.jpg'
        success = download_image(id, card_path, header)
        while not success:
            success = download_image(id, card_path, header)
        print('added', id)
        csv_data.append([id, card_path])

    with open(csv_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(csv_data)



def get_card_ids(search_uris: list, header: dict):
    card_ids = []
    while search_uris:
        uri = search_uris.pop()
        response = requests.get(uri, headers=header)
        
        if response.ok:
            data = response.json()
            if data.get('has_more') and data.get('next_page'):
                search_uris.append(data['next_page'])
            card_data = data.get('data')
            if card_data:
                ids = [card['id'] for card in card_data if 'id' in card]
                card_ids.extend(ids)
        else:
            print(f"Non-success status code: {response.status_code}")
            return None

    return card_ids


def download_set(set_code: str, img_path: str, csv_name: str, header: dict):
    url = f'https://api.scryfall.com/sets/{set_code}'
    params = {
        'format' : 'json',
        'pretty' : 'false'
    }

    response = requests.get(url, params=params, headers=header)

    if response.ok:
        data = response.json()
        search_uri = data.get('search_uri')
        if search_uri:
            card_ids = get_card_ids([search_uri], header)
            print(f'found ids #{len(card_ids)}')
            build_csv(card_ids, img_path, csv_name, header)
        else:
            print("Could Not Find Search Uri")
            return None
    else:
        print(f"Non-success status code: {response.status_code}")
        return None


header={'User-Agent' : 'MTG_Card_Scanner/1.0', 'accept' : 'application/json'}
img_path = 'images/'
csv_name = 'mtgdb.csv'
download_set('mh3', img_path, csv_name, header)