#! /bin/bash

input_file=$1
output_dir=$2


if [ ! -f "$input_file" ]; then
    echo "Error: $input_file does not exist"
    exit 1
fi

if [ ! -z ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

temp_file=${output_dir}/temp.csv
tail -n +2 $input_file | shuf > $temp_file

total_lines=$(wc -l < $temp_file | cut -d ' ' -f1)
echo "Total lines: $total_lines"
cat <(head -n 1 $input_file) <(head -n $((total_lines * 95 / 100)) $temp_file) > ${output_dir}/train.csv
cat <(head -n 1 $input_file) <(tail -n +$((total_lines * 95 / 100 + 1)) $temp_file) > ${output_dir}/test.csv
echo "Train: $(wc -l ${output_dir}/train.csv)"
echo "Test: $(wc -l ${output_dir}/test.csv)"

rm $temp_file
