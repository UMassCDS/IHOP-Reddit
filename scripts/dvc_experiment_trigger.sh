#!/bin/bash
# Hyperparam tuning using DVC - Trigger DVC experiments that test which different values of removing the top percentage of most frequently commenting users.

top_user_removal_percs=(0.02 0.05 0.10)
for turp in "${top_user_removal_percs[@]}"
do
	dvc exp run --queue -S exclude_top_users=${turp}
done

dvc exp run --run-all