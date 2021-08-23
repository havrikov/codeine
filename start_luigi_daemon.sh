#!/bin/zsh
pkill -f luigid
rm -rf ~/.luigi
luigid --background --pidfile ~/.luigi/pid --logdir ~/.luigi/logs --state-path ~/.luigi/state --port 8089
