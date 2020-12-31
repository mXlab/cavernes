#!/usr/bin/bash

if [ "$#" -ne "3" ];
then
	echo "Usage: $0 <path_to_files> <n_images> <size>"
	echo "  Example: $0 . 64 128"
	exit
fi

path=$1
nimg=$2
size=$3
tmpdir="${path}/tmp"
echo "Splitting all files"
mkdir -p $tmpdir
for file in ${path}/*.png
do
	name=$(basename "$file" ".png")
	echo "$name"
	convert -crop ${size}x${size} $file ${tmpdir}/${name}_%02d.png
done

echo "Putting files into individual folders"
for ((i=0; i<nimg; ++i));
do
	index=$(printf "%02d" $i)
	dirname="images_png_${index}"
	echo "$dirname"
	dir="${path}/${dirname}"
	mkdir -p $dir
	regexp="${tmpdir}/*_${index}.png"
	mv $regexp $dir
done
rmdir $tmpdir
