import requests
import time

import config

class EMA():
    '''Exponential moving average'''
    def __init__(self, init: float):
        self.value = init
        self.n_samples = 0

    def __iadd__(self, other: float):
        self.n_samples += 1
        self.value = (
            self.value
            - (self.value / self.n_samples)
            + other / self.n_samples)
        return self


def download(
        url: str,
        num_retries: int = config.DOWNLOAD_NUM_RETRIES) -> bytes:
    for i in range(num_retries + 1):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print('Download failed:',
                      url, response.status_code, response.reason)
            assert response.status_code == 200
            return response.content
        except requests.exceptions.RequestException as e:
            print(f'Download excepted, retrying ({i}):',
                  url)
            if i < num_retries:
                time.sleep(2 ** i)
            else:
                raise e
    assert False  # unreachable
