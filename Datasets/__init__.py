def load(
    name,
    data_path,
    dump_path,
    num_samples_per_day=1,
    tweets_per_sample=1000,
    min_df=0.05,
    max_df=0.8,
    ks=20,
    sigma=5,
    use_lds=False,
    total_samples=10000,
):

    if name == "twitter":
        from .twitter import create_dataset

        return create_dataset(
            data_path,
            num_samples_per_day,
            tweets_per_sample,
            min_df,
            max_df,
            dump_path,
            ks,
            sigma,
            use_lds,
        )
    elif name == "twitter_upsample":
        from .twitter_upsample import create_dataset

        return create_dataset(
            data_path,
            total_samples,
            tweets_per_sample,
            min_df,
            max_df,
            dump_path,
        )
    else:
        return NotImplementedError(f"Dataset {name} not implemented.")
