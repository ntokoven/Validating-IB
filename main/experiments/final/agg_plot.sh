mkdir results
mkdir results/cifar10/
mkdir results/cifar10/gc_plots

mkdir results/mnist12k/
mkdir results/mnist12k/gc_plots

###############################

i=0
for wd in 0 1e-7 1e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3
do
	#for enc in stoch determ
	#do
             	echo $wd, $i #, $enc
		echo "wd_{$i}_$wd"
                scp gencap/mnist12k/wd/$wd/stoch/acc_num_labels.png results/mnist12k/gc_plots/wd_stoch_{$i}_$wd.png
		scp gencap/mnist12k/wd/$wd/determ/acc_num_labels.png results/mnist12k/gc_plots/wd_determ_{$i}_$wd.png
        	let "i+=1"
	#done
done

###############################

i=0
for wd in  0 1e-7 1e-6 1e-5 5e-5 75e-6 1e-4
do
	#for enc in stoch determ
	#do
             	# echo $wd, $i, $enc
		echo "wd_{$i}_$wd"
                scp gencap/cifar10/wd/$wd/determ/acc_num_labels.png results/cifar10/gc_plots/wd_determ_{$i}_$wd.png
		scp gencap/cifar10/wd/$wd/stoch/acc_num_labels.png results/cifar10/gc_plots/wd_stoch_{$i}_$wd.png
        	let "i+=1"
	#done
done

###############################


i=0
for dout in 0 0.1 0.2 0.325 0.45 0.6 0.75 
do
	# for enc in stoch determ
	# do
             	# echo $dout, $i, $enc
		echo "dout_{$i}_$dout"
                scp gencap/mnist12k/dropout/$dout/stoch/acc_num_labels.png results/mnist12k/gc_plots/dout_stoch_{$i}_$dout.png
		scp gencap/mnist12k/dropout/$dout/determ/acc_num_labels.png results/mnist12k/gc_plots/dout_determ_{$i}_$dout.png
        	let "i+=1"
	# done
done

###############################

i=0
for dout in 0 0.1 0.2 0.325 0.4 0.475
do
	# for enc in stoch determ
	# do
             	# echo $dout, $i, $enc
		echo "dout_{$i}_$dout"
                scp gencap/cifar10/dropout/$dout/stoch/acc_num_labels.png results/cifar10/gc_plots/dout_stoch_{$i}_$dout.png
		scp gencap/cifar10/dropout/$dout/determ/acc_num_labels.png results/cifar10/gc_plots/dout_determ_{$i}_$dout.png
        	let "i+=1"
	# done
	
	
done

###############################

i=0
for beta in 0 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 
do
	echo $beta, $i
	scp gencap/mnist12k/vib/$beta/acc_numlabels.png results/mnist12k/gc_plots/vib_{$i}_$beta.png
	
	# for sigma in unit learnt
	# do	
		
		scp gencap/mnist12k/ceb/$beta/unit/acc_num_labels.png results/mnist12k/gc_plots/ceb_unit_{$i}_$beta.png
		scp gencap/mnist12k/ceb/$beta/learnt/acc_num_labels.png results/mnist12k/gc_plots/ceb_learnt_{$i}_$beta.png
	# done
	let "i+=1"
done

i=0
for beta in 0 1e-8 1e-7 1e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 5875e-6
do
	echo $beta, $i
	scp gencap/cifar10/vib/$beta/acc_numlabels.png results/cifar10/gc_plots/vib_{$i}_$beta.png
	
	# for sigma in unit learnt
	# do	
		
		scp gencap/cifar10/ceb/$beta/unit/acc_num_labels.png results/cifar10/gc_plots/ceb_unit_{$i}_$beta.png
		scp gencap/cifar10/ceb/$beta/learnt/acc_num_labels.png results/cifar10/gc_plots/ceb_learnt_{$i}_$beta.png
	# done
	let "i+=1"
done
