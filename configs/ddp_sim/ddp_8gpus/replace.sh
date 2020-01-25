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


# add  "\"bs_train\": 128" "\"ddp_sim_num_gpus\": 4,"
# replace "\"out_filename\": \"seq\"" "\"out_filename\": \"ddp_sim\""
replace "\"bs_train\": 512" "\"bs_train\": 1024"
replace "\"lr\": 0.4" "\"lr\": 0.8"
replace "\"ddp_sim_num_gpus\": 4" "\"ddp_sim_num_gpus\": 8"
replace "\"epochs\": 200" "\"epochs\": 400"


