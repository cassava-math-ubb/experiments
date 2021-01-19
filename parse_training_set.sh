mkdir ./flow ./flow/0 ./flow/1 ./flow/2 ./flow/3 ./flow/4;

tail -n +1 train.csv | while IFS=, read -r col1 col2
do
    echo "$col1 | $col2"
    cp ./train_images/$col1 ./flow/$col2/$col1
done;

mv ./flow/0 ./flow/Bacterial\ Blight
mv ./flow/1 ./flow/Brown\ Streak\ Disease
mv ./flow/2 ./flow/Green\ Mottle
mv ./flow/3 ./flow/Mosaic\ Disease
mv ./flow/4 ./flow/Healthy
