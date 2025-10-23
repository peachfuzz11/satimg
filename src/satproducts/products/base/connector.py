import abc
import datetime


class Connector(abc.ABC):
    def __init__(self):
        super(Connector, self).__init__()

    @abc.abstractmethod
    def download(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def search(self, geojson=None, start_datetime: datetime.datetime = None, stop_datetime: datetime.datetime = None,
               *args, **kwargs):
        pass
