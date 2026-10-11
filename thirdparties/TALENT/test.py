import optuna

# Load the study from the database
study = optuna.load_study(
    study_name="danets_mimicpipe_UTI_tuning",  # You may need to find the study name first
    storage="sqlite:////home/mpham/workspace/source/ehrfm/tabular-training/TALENT/outputs/all/hyp-tune/eicu2/UTI/danets/optuna_study.db"
)

# Get best trial
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Get all trials
trials_df = study.trials_dataframe()
print(trials_df)