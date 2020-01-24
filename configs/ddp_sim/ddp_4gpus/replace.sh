function replace {
	echo replacing $1 with $2:
	for i in *.json; do sed -i "s/${1}/${2}/g" "$i"; done
	for i in *.json; do echo "$i"; done
}

function add {
	echo adding $1 after $2:
	for i in *.json; do sed -i "/${1}/a ${2}" "$i"; done
	for i in *.json; do echo "$i"; done
}




# replace wrn_28x10_c100_dr03_p4 wrn_16x4_c100_dr03_p4 
# replace results 'results\/4partitions\/wrn16x4_cifar100'
# add  "\"logdir\": \"logs\/\"" "\"data_dir\": \"\/home_local\/saareliad\/data\","

# add  "\"bs_train\": 128" "\"ddp_sim_num_gpus\": 4,"
replace "\"out_filename\": \"seq\"" "\"out_filename\": \"ddp_sim\""
replace "\"bs_train\": 128" "\"bs_train\": 512"

# replace wrn_16x4_c100_dr03_p4 wrn_16x4_c100_p4

# replace "\"epochs\": 200" "\"epochs\": 220"

