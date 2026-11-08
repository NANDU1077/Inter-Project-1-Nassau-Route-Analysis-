# Video scripts (fill in the [brackets] with your own details)

## Video 1: Project explanation (about 7-8 minutes)

**0:00 Introduction (dashboard open).** "Hello, I'm Nanda Kishore. This is my machine learning internship project: Factory-to-Customer
Shipping Route Efficiency Analysis for Nassau Candy Distributor. I'll explain what it does, how I built it and what I found."

**0:30 Problem.** "Nassau Candy ships from five factories to customers across the US and Canada. It has detailed order data but no view of which
routes are fast or slow, where delays concentrate, or whether delays can be predicted. This project builds that view and tests whether machine
learning can predict lead time and delays."

**1:00 Data.** "10,194 order lines, 18 fields, orders from January 2024 to December 2025. Each product maps to one of five factories. A route is
factory to state or region. A shipment is a unique order and factory pair: 8,549 shipments to 5,044 customers."

**1:45 Key data finding (Data Quality page).** "Every ship date is 904 to 1,642 days after the order date, and 95 percent are in the future, so they
cannot be real shipping times. The values cluster about one year apart, which suggests a systematic date offset. That is my inference, not proven.
I did not hide it: every row is flagged, a warning shows on every page, and I created a clearly labelled relative lead time that removes the
whole-year offset so routes can be compared. As a check, Same Day averages 0.4 relative days and Standard Class 5.3, the expected order, while the
recorded values show no such pattern. I also found a calendar artifact: periods spanning a leap day sit about 0.4 days higher, so the relative
basis carries roughly one day of noise."

**3:15 Analysis method.** "I calculate shipments, average and median lead time, variability, delay frequency against a threshold slider, and a
0 to 100 route efficiency score, only for routes with at least five shipments."

**3:45 Dashboard demo.** Route Efficiency (leaderboard, top and bottom ten) → Geographic Map (US heatmap, routes, Canada) → Ship Mode (Standard vs
Expedited) → Route Drill-Down (timeline, state insights) → filters and the empty-filter message.

**5:00 Machine learning (Lead-Time & Delay Prediction page; press Train first).** "I tested whether lead time and delay can be predicted from route, ship
mode, product and order timing. I used five-fold cross-validation grouped by customer, so no customer appears in both training and test data, and I
compared every model with a baseline: the average for each ship mode. I trained Ridge regression, random forest and gradient boosting."
"Result: none of the models predicted lead time better than the baseline, which already reaches R squared of 0.62. A model using only route
information, meaning factory, state and region, has R squared of about zero. For delay classification, logistic regression raised AUC from 0.726
to 0.745, a small but statistically detectable gain. Permutation importance shows ship mode matters and everything else is near zero."
"I also grouped routes with K-means into three segments, but the silhouette score is only 0.35, so I treat that as exploratory."
"This is a useful negative result: routes differ, but not enough to predict, so a route-level prediction tool is not justified with this data."
Show the what-if predictor briefly.

**6:30 Findings and recommendations.** "Ship mode drives lead time. Factory and region differences are mostly ship-mode mix. Routes worth reviewing
first: New York and Illinois from the two large factories, plus Alberta, Minnesota and New Mexico. First step: confirm the ship-date source."

**7:15 Limitations and close.** "The dates are unreliable, there is no delivery date or freight cost, and many routes have few shipments. The code
is on GitHub, the app is deployed, and I wrote a research paper and executive summary. Thank you."

## Video 2: Internship experience and feedback (about 2-3 minutes)

**Intro (15 s).** "Hello, I'm Nanda Kishore. I completed my machine learning internship with Unified Mentor, and I'd like to share my experience."

**What I worked on (30 s).** "My project analysed shipping route efficiency for a candy distributor. I cleaned and validated the data, engineered
route features, built an interactive Streamlit dashboard, trained and cross-validated machine learning models, deployed the app, and wrote a research
paper and executive summary."

**What I learned (60 s), keep what is true for you.**
- "Real data is rarely clean. Every ship date was implausible, and I learned to report that openly instead of hiding it."
- "Model evaluation matters more than model choice. I learned to use a baseline, group cross-validation to avoid leakage, and confidence intervals
  to check whether a model is really better."
- "A model that does not beat the baseline is still a valid finding. Complex models were not better here, and saying so is the honest result."
- "I improved my skills in Python, pandas, scikit-learn, Plotly and Streamlit, plus testing and deployment."

**Challenges (30 s).** "[One real challenge, such as the deployment problems or setting up Python.] I solved it by [what you did]."

**Feedback (30 s).** "What I valued most was [the hands-on project / freedom to choose methods / mentorship]. A suggestion would be [your genuine
suggestion, e.g. clearer data documentation or a data check before starting]."

**Close (10 s).** "Thank you to the Unified Mentor team for this opportunity. I'm looking forward to [your next step]."
