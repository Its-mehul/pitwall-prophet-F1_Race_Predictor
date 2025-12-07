Prompt I used :

I am doing a machine learning class project where I predict whether a Formula 1 driver will win a race based on mid-race style features that I collected and engineered myself by scraping data from formula1.com.

My dataset is at the granularity of one row per driver per race for the 2025 season (first 15 races). Each row has:

Metadata columns (not used as numeric inputs):

year, round, race_slug, race_name, driver_code, driver_name, team_name

Numeric feature columns (used as model inputs), including things like:

grid_pos, grid_pos_norm (starting grid position and normalized)

pit_stops_int (number of pit stops)

first_pit_frac, last_pit_frac (first/last pit lap divided by total race laps)

pitted_at_all, pit_before_half, total_pit_time_sec (approx mid-race pit strategy information)

fastest_lap_rank_norm, fastest_lap_lap_frac, fastest_lap_delta (pace proxies derived from fastest lap timing within the race)

plus one-hot-encoded team columns like team_Red Bull, team_Mercedes, etc.

The label column:

is_winner ∈ {0,1}, where 1 means that driver ultimately won that race.

I want a strong but not overcomplicated baseline model implemented in PyTorch that satisfies a final project requirement to “ask ChatGPT for a baseline neural network and then go beyond it.”

Please propose a baseline that includes:

How to split the data into train / validation / test, given that I have a round column (race number).

How to select the feature columns and standardize/normalize them.

A simple feed-forward neural network (multi-layer perceptron) architecture that is appropriate for this tabular binary classification task, including layer sizes and activation functions.

The loss function and how to handle class imbalance (only one winner per race), and a reasonable choice of optimizer, learning rate, batch size, and number of epochs.

Basic evaluation metrics I should report (e.g., accuracy, F1 for the positive class, etc.).

Please describe the baseline in a clear, structured way so that I can both (a) implement it directly in code, and (b) paste your answer into my project report as “the baseline suggested by ChatGPT.


The gpt response : 

