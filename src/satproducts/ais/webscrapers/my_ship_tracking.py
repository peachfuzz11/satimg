import requests
from bs4 import BeautifulSoup

from satproducts.ais.webscrapers.webscraper import Webscraper


class MyShipTrackingScraper(Webscraper):
    base_url = 'https://www.myshiptracking.com'
    search_url = base_url + '/vessels?side=false&name='

    def scrape(self, mmsi: str):
        used_info = {
            'name': None,
            'imo': None,
            'flag': None,
            'length': None,
            'width': None,
            'dwt': None,
            'gt': None,
            'ship_type': None,
            'build': None,
            'callsign': None
        }
        scraped_info = {}

        request = requests.get(self.search_url + str(mmsi))
        soup = BeautifulSoup(request.text, 'html.parser')

        ship_url = None

        for table in soup.find_all("tbody", {'class', 'table-body'}):
            for link in table.find_all('a'):
                ship_url = link['href']
            if ship_url is not None:
                type_element = table.find("div", class_="icon-cont")
                if type_element is not None:
                    scraped_info['ship_type'] = type_element.get_text(strip=True)
                break
        if ship_url is None:
            return used_info

        request = requests.get(self.base_url + str(ship_url))
        soup = BeautifulSoup(request.text, 'html.parser')

        name_element = soup.find('h1', class_='mb-0')
        if name_element:
            scraped_info['name'] = name_element.text

        for table in soup.find_all('table', {'class', 'table'}):
            for row in table.find_all('tr'):
                key = row.find('th')
                value = row.find('td')
                if key is not None and value is not None:
                    key = key.text
                    value = value.text
                    if value == '---' or value == '--- ':
                        continue
                    scraped_info[key.lower()] = value

        for key in scraped_info.keys():
            if key == 'name':
                used_info[key] = scraped_info[key]
            elif key == 'imo':
                used_info[key] = scraped_info[key]
            elif key == 'flag':
                used_info[key] = scraped_info[key].lstrip()
            elif key == 'size':
                values = scraped_info[key].split(' ')
                length, width = values[0], values[2]
                used_info['length'] = float(length)
                used_info['width'] = float(width)
            elif key == 'dwt':
                dwt = scraped_info[key].split(' ')[0]
                used_info['dwt'] = float(dwt.replace(',', '.'))
            elif key == 'gt':
                gt = scraped_info[key].split(' ')[0]
                used_info['gt'] = float(gt.replace(',', '.'))
            elif key == 'ship_type':
                used_info['ship_type'] = scraped_info[key]
            elif key == 'call sign':
                used_info['callsign'] = scraped_info[key]
            elif key == 'build':
                used_info['build'] = scraped_info[key]

        return used_info
