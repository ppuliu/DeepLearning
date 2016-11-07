TRAIN_DIR=/home/honglei/projects/heterlearning/data/shuffle_10/
TEST_DIR=/home/honglei/projects/heterlearning/data/shuffle_10_test/
REG=$1
RATE=$2
CELL_SIZE=$3
SUFFIX=$4

python run_training.py 0 $TRAIN_DIR $REG $RATE $CELL_SIZE $SUFFIX 

python run_training.py 1 $TEST_DIR $REG $RATE $CELL_SIZE $SUFFIX

python run_training.py 2 $TEST_DIR $REG $RATE $CELL_SIZE $SUFFIX
