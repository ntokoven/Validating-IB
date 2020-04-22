for wd in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2
do
	cd $wd
	rm -r estimator_models
	echo "Deleted in " $wd
	cd ..
done
