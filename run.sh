# Set root directory (Please put the root directory here, e.g. /data/Fed-DIP)
ROOT_DIR=

export PYTHONPATH="$ROOT_DIR"

# Create directories and log file
mkdir -p $ROOT_DIR/train_log
touch $ROOT_DIR/train_log/process_ids.txt
echo "Process IDs for tuning experiments - $(date)" > $ROOT_DIR/train_log/process_ids.txt

nohup python -u $ROOT_DIR/lab/methods/fed_at_clip.py \
    --dataset pacs \
    > $ROOT_DIR/train_log/pacs.log 2>&1 &
echo "pacs, PID: $!" >> $ROOT_DIR/train_log/process_ids.txt
sleep 10

nohup python -u $ROOT_DIR/lab/methods/fed_at_clip.py \
    --dataset office_home \
    > $ROOT_DIR/train_log/office_home.log 2>&1 &
echo "office_home, PID: $!" >> $ROOT_DIR/train_log/process_ids.txt
sleep 10

nohup python -u $ROOT_DIR/lab/methods/fed_at_clip.py \
    --dataset vlcs \
    > $ROOT_DIR/train_log/vlcs.log 2>&1 &
echo "vlcs, PID: $!" >> $ROOT_DIR/train_log/process_ids.txt
sleep 10

nohup python -u $ROOT_DIR/lab/methods/fed_at_clip.py \
    --dataset domain_net \
    > $ROOT_DIR/train_log/domain_net.log 2>&1 &
echo "domain_net, PID: $!" >> $ROOT_DIR/train_log/process_ids.txt
sleep 10

# Display the process IDs
echo "All processes launched. Process IDs saved to $ROOT_DIR/train_log/process_ids.txt"
cat $ROOT_DIR/train_log/process_ids.txt



