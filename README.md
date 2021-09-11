# Forager
A rapid data exploration engine.


## Requirements
Forager runtime requires:
- Python >= 3.8

Forager build requires:
- npm >= 7.19.1

## Getting started

First, install Forager:
```bash
git clone --single-branch --branch selfserve https://github.com/jeremyephron/forager.git
pushd forager
pip3 install build
python3 -m build
pip3 install dist/*.whl
popd 
```

Now you can start up the Forager server by running:
```bash
forager-server
```

You can now access your Forager instance by typing [http://localhost:4000](http://localhost:4000) in your browser.

Note that the index page for Forager is empty. That's because we haven't loaded a dataset yet.

To do so, install the Python Forager client, foragerpy:

```bash
git clone https://github.com/Forager-Research/foragerpy.git
pushd foragerpy
poetry install
poetry build
pip3 install dist/*.whl
popd
```

To load a dataset, you can start an asyncio-enabled REPL using `python3 -m asyncio` and then run the following:

```python
import foragerpy.client
client = foragerpy.client.Client(user_email="<YOUR@EMAIL.COM>")
await client.add_dataset('<DATASET_NAME>', '/path/to/train/images/directory, '/path/to/val/images/directory')
```

Now refresh the Forager web page and you should see your new dataset.

NOTE: Forager sessions are currently ephemeral--the database is stored as a temporary file which may be cleared on reboot.

