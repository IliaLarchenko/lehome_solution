#!/bin/bash
# Kill all eval-related processes (safe to run from any context)
echo "Killing eval processes..."
pkill -9 -f "scripts.eval" 2>/dev/null
pkill -9 -f "scripts/serve.py" 2>/dev/null
pkill -9 -f "scripts/run_eval" 2>/dev/null
pkill -9 -f "scripts/eval_worker" 2>/dev/null
sleep 2
remaining=$(ps aux | grep -E "(run_eval|serve.py|scripts.eval)" | grep -v grep | grep -v kill_evals | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "All clean"
else
    echo "WARNING: $remaining processes still running:"
    ps aux | grep -E "(run_eval|serve.py|scripts.eval)" | grep -v grep | grep -v kill_evals
fi
