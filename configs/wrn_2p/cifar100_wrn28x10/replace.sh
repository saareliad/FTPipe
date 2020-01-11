function replace {
	echo replacing $1 with $2:
	for i in *.json; do sed -i "s/${1}/${2}/g" "$i"; done
	for i in *.json; do echo "$i"; done
}


replace wrn_28x10_c100_p2 wrn_28x10_c100_dr03_p2 
# replace wrn_16x4_c100_p2 wrn_28x10_c100_p2
# replace "\"epochs\": 200" "\"epochs\": 220"
# replace "\"ninja4\" \"ninja4\"" "\"out_filename\": \"ninja\""
#replace "\"out_filename\": \"ninja4\"" "\"out_filename\": \"ninja\""

