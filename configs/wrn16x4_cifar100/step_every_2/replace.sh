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




# replace wrn_28x10_c100_dr03_p4 wrn_16x4_c100_dr03_p4 
# replace results 'results\/4partitions\/wrn16x4_cifar100'
# add  "\"logdir\": \"logs\/\"" "\"data_dir\": \"\/home_local\/saareliad\/data\","


add  "\"epochs\": 220" "\"cudnn_benchmark\": true,"
add  "\"epochs\": 220" "\"step_every\": 2,"
replace  "\"epochs\": 220" "\"epochs\": 400"
replace "\"lr\": 0.1" "\"lr\": 0.2"
replace "\"out_filename\": \"exp\"" "\"out_filename\": \"bb_pipe\""

# add  "\"logdir\": \"logs\/\"" "\"data_dir\": \"\/home_local\/saareliad\/data\","


# replace wrn_16x4_c100_dr03_p4 wrn_16x4_c100_p4

# replace "\"epochs\": 200" "\"epochs\": 220"
# replace "\"out_filename\": \"ninja4\"" "\"out_filename\": \"ninja\""

