import requests
from bs4 import BeautifulSoup

from satproducts.ais.webscrapers.webscraper import Webscraper


class VesselFinderScraper(Webscraper):
    base_url = 'https://www.vesselfinder.com'
    search_url = base_url + '/vessels/details/'

    def scrape(self, mmsi: str):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
        }

        used_info = {'imo': None, 'name': None, 'callsign': None, 'ship_type': None, 'ais_type': None, 'build': None,
                     'length': None, 'width': None, 'gt': None, 'dwt': None, 'flag': None}
        scraped_info = {}

        request = requests.get(self.search_url + str(mmsi), headers=headers)
        soup = BeautifulSoup(request.text, 'html.parser')

        name_element = soup.find("h1", class_="title")
        if name_element:
            scraped_info['name'] = name_element.text

        vessel_particulars_header = soup.find("h2", class_="bar", string="Vessel Particulars")
        voyage_data_header = soup.find("h2", class_="bar2", string="Voyage Data")

        if vessel_particulars_header is not None:
            tables = vessel_particulars_header.find_next_siblings()
            for table in tables:
                for row in table.find_all('tr'):
                    key = row.find('td', class_='tpc1')
                    value = row.find('td', class_='tpc2')
                    if key is not None and value is not None:
                        if value.text == '' or value.text == '-':
                            continue
                        key = key.text.lower()
                        value = value.text
                        scraped_info[key] = value
        if voyage_data_header is not None:
            tables = voyage_data_header.find_next_siblings()
            for table in tables:
                for row in table.find_all('tr'):
                    key = row.find('td', class_='n3')
                    value = row.find('td', class_='v3')
                    if key is not None and value is not None:
                        if value.text == '' or value.text == '-':
                            continue
                        key = key.text.lower()
                        value = value.text
                        scraped_info[key] = value

        for key in scraped_info.keys():
            if key == 'imo number':
                used_info['imo'] = scraped_info[key]
            elif key == 'imo / mmsi' and vessel_particulars_header is None:
                values = scraped_info[key].split(' ')
                used_info['imo'] = values[0]
            elif key == 'name':
                used_info['name'] = scraped_info[key]
            elif key == 'callsign':
                used_info['callsign'] = scraped_info[key]
            elif key == 'ship type':
                used_info['ship_type'] = scraped_info[key]
            elif key == 'ais type':
                used_info['ais_type'] = scraped_info[key]
            elif key == 'year of build':
                used_info['build'] = scraped_info[key]
            elif key == 'length / beam' and vessel_particulars_header is None:
                values = scraped_info[key].split(' ')
                used_info['length'] = float(values[0])
                used_info['width'] = float(values[2])
            elif key == 'length overall (m)':
                used_info['length'] = float(scraped_info[key])
            elif key == 'beam (m)':
                used_info['width'] = float(scraped_info[key])
            elif key == 'gross tonnage':
                used_info['gt'] = float(scraped_info[key])
            elif key == 'dwt':
                used_info['dwt'] = float(scraped_info[key])
            elif key == 'flag':
                used_info['flag'] = scraped_info[key]

        return used_info
