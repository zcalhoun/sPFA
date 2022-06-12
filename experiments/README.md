# Experiments
Descriptions of the experiments run are provided below, along with any conclusions derived from each experiment.

## Test experiments
The files that start with `test` are used if any significant change is made to the model, to ensure that a model can run without encountering any errors.

## 1000_10_by_file
The raw data has already been lemmatized, so that process does not need to occur every time the script runs. The lemmatized tweets are organized into files with the format `{city}_{date}.json`. For each of these pre-lemmatized files, 1000 tweets are randomly sampled from each city_day file and summed to create one representation for that day. The second number, 10, refers to the number of samples drawn from each city_day file.

The idea with these experiments is to determine whether sampling strategy matters. Each city has a tweet density, and a city like Los Angeles is going to have far more tweets on any given day than a smaller city like Raleigh. So, with this strategy, we want to get the variability within a city, without giving the model too many samples for any one city.

Since the sampling strategies repeat themselves, the sampled data is saved in an intermediate folder so that subsequent experiments using this strategy can simply load the files that have already been generated.

## 1000_10_by_vol
The alternative approach to randomly drawing 10 sets of 1000 tweets from each city_day combination is to sample such that each tweet individually is sampled 10 times. This approach ensures that all of the data is used from each city_day combination. However, the intuition here is that this means larger cities with higher tweet densities may be overrepresented. Nonetheless, this provides more variability within those larger cities so that those larger cities can be represented well, too.

Bash files prepended with 1000_10_by_vol are meant to test this approach. The number 10 is arbitrarily selected in this case, but the number was found experimentally to provide enough permutations of the data to give some sense of the distribution of topics for that day.

This approach will be more memory intensive, as there will be signifantly more examples to look at. The goal is to determine whether a better model is learned from this approach than by merely sampling a set number of times from each city_day.

## Experiments
| Experiment Number | File | Results Folder |
|:------------------|:----:|:--------------:|
| 1 | [1000_10_by_file_1.sh](1000_10_by_file_1.sh) | [../results/exp1](../results/exp1)|
| 2 | [1000_10_by_vol_1.sh](1000_10_by_vol_1.sh) | [../results/exp2](../results/exp2)|

