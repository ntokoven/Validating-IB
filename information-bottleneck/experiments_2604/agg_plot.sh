i=0
for wd in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2
do
             	echo $wd, $i
		echo "wd_{$i}_$wd"
                scp gencap/wd_whitened/$wd/acc_numlabels.png results/gc_plots/wd_{$i}_$wd.png
        	let "i+=1"

done


i=0
for dout in 0 0.05 0.1 0.2 0.325 0.45 0.6 0.75 0.9
do 
        		
		echo $dout, $i
                scp gencap/dropout/$dout/acc_numlabels.png results/gc_plots/dout_{$i}_$dout.png
        	let "i+=1"
done

i=0
for vib in 0 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e-0 
do
        
		echo $vib, $i
                scp gencap/vib/$vib/acc_numlabels.png results/gc_plots/vib_{$i}_$vib.png
      		let "i+=1"
done
