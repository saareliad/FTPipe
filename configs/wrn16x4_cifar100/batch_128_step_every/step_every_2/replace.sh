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


# add  "\"epochs\": 220" "\"cudnn_benchmark\": true,"
# add  "\"epochs\": 220" "\"step_every\": 2,"
# # replace  "\"epochs\": 220" "\"epochs\": 400"
# # replace "\"lr\": 0.1" "\"lr\": 0.2"
# replace "\"out_filename\": \"exp\"" "\"out_filename\": \"agg\""

replace "\"bs_train\": 128" "\"bs_train\": 64" 

