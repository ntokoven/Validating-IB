mkdir results
mkdir results/abc_values
mkdir results/abc_values/$1

i=0
for vib in 0 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 5e-2 1e-1 5e-1 1 5 
do
        
		echo $vib, $i
                scp gencap/$1/ceb/$vib/abc_values.csv results/abc_values/$1/ceb_{$i}_$vib.csv
		scp gencap/$1/vib/$vib/abc_values.csv results/abc_values/$1/vib_{$i}_$vib.csv
      		let "i+=1"
done
