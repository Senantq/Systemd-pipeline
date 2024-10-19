#!/bin/bash

# Doit être exécuté en root uniquement

# Répertoire où se trouvent les fichiers .service
service_dir="$(dirname "$0")"

# Parcourir tous les fichiers .service dans le répertoire du script
for service_file in "$service_dir"/*.service; do
    service_name=$(basename "$service_file")

    # Exclure cleanup.service
    if [[ "$service_name" == "cleanup.service" ]]; then
        continue
    fi

    # Désactiver le service
    systemctl disable "$service_name"

    # Supprimer le fichier de /etc/systemd/system/
    rm "/etc/systemd/system/$service_name"
done

# Recharger systemd
systemctl daemon-reload
