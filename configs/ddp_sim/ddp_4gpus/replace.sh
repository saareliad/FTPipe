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
# replace "\"bs_train\": 128" "\"bs_train\": 512"
replace "\"lr\": 0.1" "\"lr\": 0.4"
# replace "\"epochs\": 200" "\"epochs\": 220"


