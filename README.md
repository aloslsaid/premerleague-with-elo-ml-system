# elo system
A machine learning model that predicts English Premier League match outcomes using team strength and recent performance. An ELO rating is calculated and updated after every match, while each team’s average goal difference from its previous five matches is extracted. These features are used to train multiple ML algorithms and predict 
EPL Match Prediction with ELO

This project is an improved version of my Premier League match prediction project.

The main idea was to add an ELO rating system to represent the strength of each team.

I started with historical Premier League match data.

Before training the model, I cleaned and prepared the data.

I wanted the features to represent information that would actually be known before each match.

One of the main features I added was the ELO rating of each team.

Each team starts with an initial rating.

After every match, the ratings are updated based on the result.

A team gains rating when it performs better than expected.

A team loses rating when it performs worse than expected.

This creates a changing rating that represents the team’s strength over time.

I also used recent team performance and other match statistics as features.

For example, I used information from previous matches such as goal difference and recent form.

The important part was updating the features in chronological order.

I did not want future matches to affect predictions for earlier matches.

After preparing the features, I separated the target from the input data.

I then trained machine learning models using the generated features.

I compared the results to understand whether the additional ELO information helped.

This project taught me more about feature engineering than simply using columns from an existing dataset.

I had to create a feature that changes after every match.

It also made me think more carefully about data leakage and time-based validation.

The ELO system gives the model a simple way to understand team strength.

The project is still experimental, but it is closer to a real predictive system than using only raw match statistics.

I used

Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

Future Improvements

I would like to optimize the ELO parameters, test more machine learning models, add more historical features, and compare the system with and without ELO.

I would also like to build an application that predicts upcoming Premier League matches.
