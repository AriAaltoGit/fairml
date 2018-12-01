import matplotlib
import pickle

# temporary work around down to virtualenv
# matplotlib issue.
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.linear_model import LogisticRegression

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies

plt.style.use('ggplot')
plt.figure(figsize=(6, 6))

# read in propublica data
#propublica_data = pd.read_csv("./doc/example_notebooks/"
#                              "propublica_data_for_fairml.csv")

# quick data processing
#compas_rating = propublica_data.score_factor.values
#propublica_data = propublica_data.drop("score_factor", 1)

#  quick setup of Logistic regression
#  perhaps use a more crazy classifier
#clf = LogisticRegression(penalty='l2', C=0.01)
#clf.fit(propublica_data.values, compas_rating)

# Open saved model
with open('../BlackBoxAuditing/BBA_model', 'rb') as model_file:
    clf = pickle.load(model_file)
# Open saved data
german_data = pd.read_csv("../BlackBoxAuditing/adult_train_SVM_matrix.csv",sep=" ", header=None)

german_data = german_data.loc[0:100]

german_test_data = pd.read_csv("../BlackBoxAuditing/adult_test_matrix.csv",sep=" ", header=None)
#print(german_data)
#exit(0)

#  call audit model
#importancies, _ = audit_model(clf.predict, german_data, external_data_set=german_test_data, number_of_runs=10)
importancies, _ = audit_model(clf.predict, german_data, number_of_runs=1)

# print feature importance
print(importancies)

# generate feature dependence plot
fig = plot_dependencies(
    importancies.median(),
    reverse_values=False,
    title="FairML feature dependence SVM model"
)

file_name = "fairml_propublica_linear_direct.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)
