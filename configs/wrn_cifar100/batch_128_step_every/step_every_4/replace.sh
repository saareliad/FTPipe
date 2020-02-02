function replace {
	echo replacing $1 with $2:
	for i in *.json; do sed -i "s/${1}/${2}/g" "$i"; done
	for i in *.json; do echo "$i"; done
}

function add {
	echo replacing $1 with $2:
	for i in *.json; do sed -i "/${1}/a ${2}" "$i"; done
	for i in *.json; do echo "$i"; done
}




replace "\"model\": \"wrn_16x4_c100_p4\"" "\"model\": \"wrn_28x10_c100_dr03_p4\"" 
replace "wrn16x4_cifar100" "wrn28x10_cifar100"

# add  "\"epochs\": 220" "\"cudnn_benchmark\": true,"
# add  "\"epochs\": 220" "\"step_every\": 4,"
# # replace  "\"epochs\": 220" "\"epochs\": 400"
# # replace "\"lr\": 0.1" "\"lr\": 0.2"
# replace "\"out_filename\": \"exp\"" "\"out_filename\": \"agg\""

# replace "\"bs_train\": 128" "\"bs_train\": 32" 



