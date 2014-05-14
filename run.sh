#!/bin/bash

usage ()
{
	echo "Usage: $0 [-i <string>] [-t <string>] [-f <string>] [-o<string>] -p -d" 1>&2; 
	echo "-i <input file> is required"
	echo "-t <file with multi tests> is required or -f"
	echo "-f <file with single test> is required or -t"
	echo "-o <directory where it will be saved results>"
	echo "-d this print all information"
	echo "-p this print data in and out"
	exit -1;
}

pflag=""
dflag=""
o=""

while getopts ":i:t:f:o:pd" opt; do
	case "${opt}" in
		i)
			input=${OPTARG}
	                ;;
		o)
			o=${OPTARG}
			;;
	        t)
			t=${OPTARG}
	                ;;
		f)
			f=${OPTARG}
			;;
		p)
			pflag="-p"
			;;
		d)
			dflag="-d"
			;;
		*)
		        usage
	                ;;
	esac
done

shift $((OPTIND-1))

if [ -z "${input}" ]; then
	usage
fi

if [ -z "${t}" ] && [ -z "${f}" ]; then
	usage
fi

if [[ ! -z "${t}" ]] && [[ ! -z "${f}" ]]; then
        usage
fi

# Check input file and if is necessary compile file
if [[ ! -f $input ]]; then
	echo "File $input does not exist."
	exit 1
fi
ext=${input#*.}
output="${input%.*}.out"

echo 
echo "Compile file $input output $output"
case "$ext" in
        c) 	echo "C"
		gcc $input -o $output
          	;;
        cpp) 	echo "C++"
		g++ $input -o $output
          ;;
        cu) 	echo "Cuda"
		nvcc -arch=sm_20 $input -o $output
	  ;;
esac

printf '\n################ Run test ##############\n\n'

# IF run the multi test
if [[ ! -z "${t}" ]]; then

	if [[ ! -f $t ]]; then
		echo "Test $t does not exist."
		exit 1
	fi
	dir=$(dirname ${t})


	while read line           
	do
		echo "Begin test - $line"	
		if [ "${o}" == "" ]; then
			./$output $pflag $dflag < $dir/$line
		else
			./$output $pflag $dflag < $dir/$line > "$o/$line"
		fi
		echo "End test - $line"
		echo
	done < $t 
	exit 1
fi


# IF run single test
base=$(basename $f)
echo "Begin test - $base" 
if [ "${o}" == "" ]; then
	./$output $pflag $dflag < $f
else
	./$output $pflag $dflag < $f > "$o/$base"
fi
echo "End test - $base"


