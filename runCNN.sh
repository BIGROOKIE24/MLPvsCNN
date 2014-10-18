#!/usr/bin/sh
script=$1
benchmark=$2
height=$3
width=$4
denoiserAE=$5
n_epochs=200
batch_size=500
nkerns0=20
nkerns1=50
filtering=5
poolsize=2
for l_rate in "0.1" "0.2" "0.3" "0.4" "0.5"
do
    for hidden_size in {300..1100..200}
    do
#	eval "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python $script $l_rate $n_epochs $batch_size $nkerns0 $nkerns1 $filtering $poolsize $height $width $hidden_size $denoiserAE $benchmark" > exp-logs/exp.log.$l_rate.$hidden_size 2> exp-logs/exp.err.$l_rate.$hidden_size
	echo "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python $script $l_rate $n_epochs $batch_size $nkerns0 $nkerns1 $filtering $poolsize $height $width $hidden_size $denoiserAE $benchmark"
    done
done
