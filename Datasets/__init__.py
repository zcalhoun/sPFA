import twitter


def load(
    data_path,
    dump_path,
    num_samples_per_day=1,
    tweets_per_sample=10000,
    min_df=0.05,
    max_df=0.8,
):
    return twitter.create_dataset(
        data_path, num_samples_per_day, tweets_per_sample, min_df, max_df, dump_path
    )
