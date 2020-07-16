mkdir results
mkdir results/eval_plots

i=0
for n in 0 1 2 4 8 16 32 64
do
        
		echo $n
                scp gencap/$1/vib/1/$n/acc_numlabels.png results/eval_plots/vib_{$i}_$n.png
		scp gencap/$1/ceb/1/$n/acc_numlabels.png results/eval_plots/ceb_{$i}_$n.png

      		let "i+=1"
done
