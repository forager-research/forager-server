# Forager
A rapid data exploration engine.

## Getting started
```bash
pip3 install forager_server
```

Now you can start up the Forager server by running:
```bash
forager-server
```

You can now access your Forager instance by typing [http://localhost:4000](http://localhost:4000) in your browser.

## Contributing

### Requirements
Forager runtime requires:
- Python >= 3.8

Forager build requires:
- npm >= 7.19.1

### Development setup

To build Forager:
```bash
git clone https://github.com/forager-research/forager-server.git
cd forager-server
pip3 install build
python3 -m build
pip3 install dist/*.whl
```
