#!/bin/bash

# 1°) Parcourir tous les fichiers .service du wd
for service_file in *.service; do
    # 2°) Les copier /etc/systemd/system/
    sudo cp "$service_file" /etc/systemd/system/
    # 3°) Les activer tous
    sudo systemctl enable "$(basename "$service_file")"
done

# 4°) Reload systemd pour bien tout prendre en compte
sudo systemctl daemon-reload

# 5°) (Re)Démarrer le premier service
sudo systemctl restart premier_script.service &

#6°) print le statut du premier
sudo systemctl status --no-pager premier_script.service
