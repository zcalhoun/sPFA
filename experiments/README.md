# Experiments
Descriptions of the experiments run are provided below, along with any conclusions derived from each experiment.

## Test experiments
The files that start with `test` are used if any significant change is made to the model, to ensure that a model can run without encountering any errors.

## 1000_10_by_file
The raw data has already been lemmatized, so that process does not need to occur every time the script runs. The lemmatized tweets are organized into files with the format `{city}_{date}.json`. For each of these pre-lemmatized files, 1000 tweets are randomly sampled from each city_day file and summed to create one representation for that day. The second number, 10, refers to the number of samples drawn from each city_day file.

The idea with these experiments is to determine whether sampling strategy matters. Each city has a tweet density, and a city like Los Angeles is going to have far more tweets on any given day than a smaller city like Raleigh. So, with this strategy, we want to get the variability within a city, without giving the model too many samples for any one city.

Since the sampling strategies repeat themselves, the sampled data is saved in an intermediate folder so that subsequent experiments using this strategy can simply load the files that have already been generated.

