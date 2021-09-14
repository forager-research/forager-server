import asyncio
import config
from handler import BGSplittingMapper

class DummyRequest():
    def __init__(self, body):
        self.body = body
        self.json = body

    async def receive_body(self):
        pass


async def main():
    request_body = {"job_id": "5aea00ec-d0a3-409c-9802-cc44c581c4f2",
                    "job_args": {"input_bucket": "foragerml", "return_type": "save",
                                 "checkpoint_path": "TEST"},
                    "inputs": [{"path": "waymo/train/1507306154061139_front.jpeg", "id": 7808}, {"path": "waymo/train/1550100917074183_front.jpeg", "id": 7809}]}
    request = DummyRequest(request_body)
    mapper = BGSplittingMapper()
    config.DATA_FILE_TMPL = './dnn_outputs/{}/{}/{}-{{}}.npy'
    await mapper._handle_request(request)


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
