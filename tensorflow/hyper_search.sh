REGS="0.1 0.01"
RATES="0.5"
CELL_SIZES="128 256"

i=20

for reg in $REGS
do
        for rate in $RATES
        do
                for size in $CELL_SIZES
                do
                        echo "====================" $i
                        bash run_training.sh $reg $rate $size $i > ../../../log_dir/log_$i.txt
                        i=$(($i + 1))
                done
        done
done
exit 0
