#!/bin/bash
# Script to run Persian OCR training in the background with nohup

LOG_FILE="train.log"

echo "=================================================================="
echo "  Starting OCR Training in Background with nohup"
echo "=================================================================="
echo "[*] Log file: $LOG_FILE"

# Run with python -u (unbuffered) so logs are written immediately to train.log
nohup python -u train.py "$@" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "[+] Training started successfully!"
echo "    - Process ID (PID) : $TRAIN_PID"
echo "    - Log Output File  : $LOG_FILE"
echo ""
echo "Commands to monitor and control:"
echo "  1. View live logs in real-time:"
echo "     tail -f $LOG_FILE"
echo ""
echo "  2. Check process status:"
echo "     ps aux | grep train.py"
echo ""
echo "  3. Stop training:"
echo "     kill $TRAIN_PID"
echo "=================================================================="
