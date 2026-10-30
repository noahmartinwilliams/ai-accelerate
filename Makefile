# For downloading and preprocessing data.

error_vs_mini_batch_number.png: data/food-calories/weight_loss.csv data/mnist/mnist-dataset.zip
	stack test
	./plotter.m
	echo "done"

data/food-calories/weight_loss.csv: data/food-calories/food-calories.zip
	unzip data/food-calories/food-calories.zip
	mv fast_food_cheat_meal_workout_burn.csv data/food-calories/
	cat data/food-calories/fast_food_cheat_meal_workout_burn.csv | sed '1d' | sed 's/\(.*\)"\(.*\),\(.*\)"\(.*\)/\1"\2 \3"\4/g' | cut -f '14,4,5,6,7,8,9,10,11,13,16,17,18,19,20,21' -d ',' > data/food-calories/weight_loss.csv
	
data/food-calories/food-calories.zip:
	mkdir -p data/food-calories || true
	curl -L -o ./data/food-calories/food-calories.zip https://www.kaggle.com/api/v1/datasets/download/samartalwar/fast-food-calorie-shock-food-vs-workout-burn
	# Acquired from here: https://www.kaggle.com/datasets/samartalwar/fast-food-calorie-shock-food-vs-workout-burn/data?select=fast_food_cheat_meal_workout_burn.csv
	# Uploaded by Samar Talwar


data/mnist/mnist-dataset.zip:
	mkdir data/mnist/
	curl -L -o ./data/mnist/mnist-dataset.zip https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
	unzip data/mnist/mnist-dataset.zip
	mv train-* data/mnist/
	mv t10k-* data/mnist/
	# Acquired from here: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-images-idx3-ubyte
	# Uploaded by Hojjat Khodabakhsh
	# Made by:
	# Yann LeCun, Courant Institute, NYU
	# Corinna Cortes, Google Labs, New York
	# Christopher J.C. Burges, Microsoft Research, Redmond

.PHONY:
clean:
	rm -rf data || true
