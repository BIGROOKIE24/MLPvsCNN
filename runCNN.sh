#!/usr/bin/sh
script=$1
act=$2
network_type=$3
n_epochs=200
batch_size=500
nkerns0=20
nkerns1=50
filtering=5
poolsize=2
benchmark=cifar10small.pkl.gz

for l_rate in "0.01"
do
    for hidden_size in 900 #{900..500..200}
    do
	for nkerns0 in 50 20
	do
	    for nkerns1 in 50 20
	    do
		for filter in 5
		do
	     	    echo "python $script --lrate $l_rate --b_size 200 --k_size0 $nkerns0 --k_size1 $nkerns1 --filter $filtering --channel 3 --height 32 --width 32 --hidden $hidden_size --benchmark $benchmark --activation $act $network_type"
#		    python $script --lrate $l_rate --b_size 200 --k_size0 $nkerns0 --k_size1 $nkerns1 --filter $filtering --channel 3 --height 32 --width 32 --hidden $hidden_size --benchmark $benchmark --activation $act $network_type > exp-logs/log/exp-log-$l_rate-$hidden_size-$nkerns0-$nkerns1-$filtering-$act 2> exp-logs/err/exp-err-$l_rate-$hidden_size-$nkerns0-$nkerns1-$filtering-$act
		done
	    done
	done
    done
done

