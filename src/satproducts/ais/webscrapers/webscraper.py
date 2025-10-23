import abc


class Webscraper(abc.ABC):
    base_url: str
    search_url: str

    @abc.abstractmethod
    def scrape(self, mmsi: str) -> dict:
        pass
