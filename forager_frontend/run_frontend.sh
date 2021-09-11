#!/usr/bin/env bash

ip_addr=$(curl -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
if [ $(id -u) = 0 ]; then
    config_user=$SUDO_USER
else
    config_user=$USER
fi

export REACT_APP_SERVER_URL=http://$ip_addr:8000
export PORT=$(jq ".frontend_port//4000" $(eval echo ~$config_user)/forager/django_settings.json)

npm run build
serve -s build
