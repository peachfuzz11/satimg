import requests
from bs4 import BeautifulSoup

from satproducts.ais.webscrapers.webscraper import Webscraper


class VesselTrackerScraper(Webscraper):
    base_url = "https://www.vesseltracker.com"
    search_url = base_url + "/en/vessels.html?term="

    def scrape(self, mmsi: str):
        used_info = {
            "name": None,
            "imo": None,
            "length": None,
            "width": None,
            "dwt": None,
            "gt": None,
            "callsign": None,
            "flag": None,
            "ais_type": None,
            "ship_type": None,
            "build": None,
        }
        scraped_info = {}

        request = requests.get(self.search_url + str(mmsi))
        soup = BeautifulSoup(request.text, "html.parser")

        ship_url = None
        for table in soup.find_all("div", {"class", "results-table"}):
            for row in table.find_all("div", {"class": "row"}):
                ship_url = row.find("a")["href"]
                break
            if ship_url is not None:
                break
        if ship_url is None:
            return used_info

        request = requests.get(self.base_url + str(ship_url))
        soup = BeautifulSoup(request.text, "html.parser")

        name_element = soup.find("h1")
        if name_element:
            scraped_info["name"] = name_element.text
        for table in soup.find_all("div", {"class", "key-value-table"}):
            for row in table.find_all("div", {"class", "row"}):
                key = row.find("div", {"class", "key"})
                value = row.find("div", {"class", "value"})
                if key is not None and value is not None:
                    if value.text.strip() == "":
                        continue
                    key = key.text.split(":")[0]
                    value = value.text
                    scraped_info[key.lower()] = value

        for key in scraped_info.keys():
            if key == "name":
                used_info["name"] = scraped_info[key]
            elif key == "imo":
                used_info["imo"] = scraped_info[key].strip()
            elif key == "length":
                length = scraped_info[key].split(" ")[0]
                used_info["length"] = float(length)
            elif key == "width":
                width = scraped_info[key].split(" ")[0]
                used_info["width"] = float(width)
            elif key == "deadweight":
                dwt = scraped_info[key].split(" ")[0]
                used_info["dwt"] = float(dwt.replace(",", "."))
            elif key == "gross tonnage":
                gt = scraped_info[key].split(" ")[0]
                used_info["gt"] = float(gt.replace(",", "."))
            elif key == "callsign":
                used_info["callsign"] = scraped_info[key]
            elif key == "flag":
                used_info["flag"] = scraped_info[key]
            elif key == "ais type":
                used_info["ais_type"] = scraped_info[key]
            elif key == "ship type":
                used_info["ship_type"] = scraped_info[key]
            elif key == "year of build":
                used_info["build"] = scraped_info[key]

        return used_info
