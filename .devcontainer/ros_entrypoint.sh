#!/bin/bash
set -e

# source "${MY_ROS}/setup.bash"

# Execute the passed command, or start bash if no command is provided
if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi