REGS="0.0"
RATES="0.1 0.5"
CELL_SIZES="128"

i=1

for reg in $REGS
do
        for rate in $RATES
        do
                for size in $CELL_SIZES
                do
                        echo "====================" $i
                        bash run_training.sh $reg $rate $size $i > ../../../log_dir/log_$i.txt
                        #python run_training.py 3 ../../../data/small $reg $rate $size $i > ../../../log_dir/log_$i.txt
                        i=$(($i + 1))
                done
        done
done
exit 0
