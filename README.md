# Systemd Pipeline Guide

To create a systemd pipeline:

First, gather all the Python scripts in one folder. In another folder, place all the scripts specific to the pipeline.

→ For each `.py` script to run, you need a `.sh` file. Its role is to source Conda, activate the Conda execution environment, and finally run the script.

→ For each Python script, you also need a `.service` file:

  → The first `.service` will run the first script, and the second will run after the successful execution of the first Python script (i.e., if the first script did not generate any errors preventing its proper execution). The second script is triggered using an `OnSuccess` variable. In the meantime, it redirects the error output and standard output of the first script.

  → The second service does the same thing. In this example, it is also the last `.service` file. It then executes a `cleanup.service` file, which must be the last service executed, and it should be run as root. To ensure correct execution order, script n°n depends on an "After" variable, in the same way as the `cleanup.service` file.

→ The `submit_services` file must be executed as sudo/root. It takes all the `.service` files from the directory where it resides and copies them into the systemd service execution folder. It also prints the execution status of the first script, which can be obtained with the command `sudo systemctl status first_script.service`.

→ The `cleanup.sh` file is called by the `cleanup.service` file, which must always be called by the last script of the pipeline. The role of `cleanup.sh` is to delete all service files copied into the systemd directory, except itself. It does this by referencing the `.service` files found in the directory where it is located.

### Warning:
When creating a new pipeline, pay close attention to the "OnSuccess" and "After" variables. The pipeline has successfully completed if `cleanup.service` was recently executed (if not, run `sudo ./cleanup.sh` in case of an error and the entire pipeline needs to be restarted).
