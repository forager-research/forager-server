Embedding Server
================

Run Instructions
----------------

- Change to user `fpoms` with `sudo -u fpoms bash`
- Connect to the embedding tmux window if it exists `tmux -S /tmp/forager a -t embedding`.
  If it doesn't exist, create it with `tmux -S /tmp/forager new -s embedding`
- Run `poetry run python run.py`
- Logging is written to `embedding_server.log` in this directory.
  You can stream it with `tail -f embedding_server.log`.
