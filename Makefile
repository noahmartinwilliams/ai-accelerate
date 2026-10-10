# For downloading and preprocessing data.

all: data/food-calories/weight_loss.csv
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

.PHONY:
clean:
	rm -rf data || true
