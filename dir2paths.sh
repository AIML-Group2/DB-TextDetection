find ./datasets/train_images -type f \( -iname '*.jpg' -o -iname '*.png' \)> ./datasets/train_imgs.txt
find ./datasets/train_gts -type f -name '*.txt' > ./datasets/train_gts.txt
paste ./datasets/train_imgs.txt ./datasets/train_gts.txt > ./datasets/train.txt

find ./datasets/test_images -type f \( -iname '*.jpg' -o -iname '*.png' \)> ./datasets/validate_imgs.txt
find ./datasets/test_gts -type f -name '*.txt' > ./datasets/validate_gts.txt
paste ./datasets/validate_imgs.txt ./datasets/validate_gts.txt > ./datasets/validate.txt

rm ./datasets/train_imgs.txt
rm ./datasets/train_gts.txt
rm ./datasets/validate_imgs.txt
rm ./datasets/validate_gts.txt
