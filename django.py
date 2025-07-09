# -*- coding: utf-8 -*-


!pip install git+https://github.com/microsoft/dowhy.git
import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd

# Data files are uploaded
from google.colab import files
uploaded = files.upload()

# Read Data file
df = pd.read_csv('django.csv')

# Import data from Github



#df = pd.read_csv(url, sep=",")

print(df.shape)
df.head()

df.describe()

import graphviz
!apt install libgraphviz-dev
!pip install pygraphviz

df.columns

df1=df[['nd','la','ld', 'lt', 'age','nuc', 'entrophy','ndev','exp', 'rexp','sexp','nf','ns','days_to_first_fix']]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df_scaled = ss.fit_transform(df1)

data_scaled_df = pd.DataFrame(df_scaled, columns = df1.columns)
data_scaled_df.head()

print(data_scaled_df.shape)

# df1 = df1.dropna()
# To replace NAN with mean values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Or use other strategies like 'median'
df1 = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

# Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io


# Replace NAN with mean values in data_scaled_df before applying pc function
imputer = SimpleImputer(strategy='mean')  # Or use other strategies like 'median'
data_scaled_df_imputed = pd.DataFrame(imputer.fit_transform(data_scaled_df), columns=data_scaled_df.columns)

labels = [f'{col}' for i, col in enumerate(data_scaled_df_imputed.columns)]
data = data_scaled_df_imputed.to_numpy() # use the imputed data

cg = pc(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(cg.G, labels=labels)
pyd.write_png('img_causal_PC_Django.png')
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

"""GES algorithm"""

import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io

# To label data

labels = [f'{col}' for i, col in enumerate(data_scaled_df_imputed.columns)]
data = data_scaled_df_imputed.to_numpy()

# Run GES with BIC score

record_bic = ges(data_scaled_df_imputed, score_func='local_score_BIC')

# Visualization
def plot_graph(record, title):
    #pyd = GraphUtils.to_pydot(record['G'])
    pyd = GraphUtils.to_pydot(record['G'], labels=labels)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = plt.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.title(title)
    plt.show()

plot_graph(record_bic, 'GES with BIC Score')
#plot_graph(record_gen, 'GES with Generalized Score')

df1=df[['nuc','entrophy','ndev','exp', 'rexp','sexp','nf','ns','days_to_first_fix']]

"""Step 1: Define causal model In the first step, we need to define a so-called structural causal model (SCM), which is a combination of the causal graph and the underlying generative models describing the data generation proces"""

import networkx as nx

causal_graph = nx.DiGraph([('sexp', 'days_to_first_fix'),
                           ('nf', 'ns'),
                           #('entrophy','ndev'),
                           ('nf', 'nuc'),
                           ('nuc', 'days_to_first_fix'),
                           ('entrophy','exp'),
                           ('rexp', 'days_to_first_fix'),
                           ('ndev', 'days_to_first_fix'),
                           ('ns', 'entrophy'),
                           ('rexp','days_to_first_fix'),
                           ('entrophy','nuc'),
                            ('entrophy','ndev'),
                           ('entrophy','exp'),
                           ('ndev','nuc'),
                           ('ndev','rexp'),
                           ('entrophy', 'days_to_first_fix'),
                           ('exp', 'sexp')])

import networkx as nx
from dowhy import CausalModel

# Impute missing values in df1 before creating the CausalModel
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Or use other strategies like 'median'
df1 = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

# Replace inf with large finite values
df1 = df1.replace([np.inf, -np.inf], np.nan)
df1 = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)


# Create a copy of the causal graph without the causal mechanisms
causal_graph_for_gml = causal_graph.copy()
for node in causal_graph_for_gml.nodes:
    causal_graph_for_gml.nodes[node].clear()  # Remove node attributes
for u, v in causal_graph_for_gml.edges:
    causal_graph_for_gml.edges[u, v].clear()  # Remove edge attributes

# Convert the cleaned causal_graph to a GML string
graph_gml = "\n".join(nx.generate_gml(causal_graph_for_gml))

# With graph
model = CausalModel(
    data=df1,
    treatment="ndev",   #ndev, #nuc,#sexp, #rexp, #ns, #nf
    outcome="days_to_first_fix",
    graph=graph_gml  # Pass the GML string instead of the DiGraph object
)

model.view_model()

"""Fitting the SCM to Data. With the data at hand and the graph constructed earlier, we can now evaluate the performance of DAG:"""

from dowhy import gcm
# Create and fit your causal model
causal_model = gcm.StructuralCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(causal_model, data_scaled_df_imputed)
gcm.fit(causal_model, data_scaled_df_imputed)

# Evaluate the causal model
evaluation_results = gcm.evaluate_causal_model(causal_model, data_scaled_df_imputed)

# Print the evaluation results
print(evaluation_results)

#Arrow Strength
import numpy as np
from dowhy.gcm.util.plotting import plot # Import the plot function

# Note: The percentage conversion only makes sense for purely positive attributions.
def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}


# Use causal_model instead of scm
arrow_strengths = gcm.arrow_strength(causal_model, target_node='days_to_first_fix')

# Assuming 'plot' is defined elsewhere and takes these arguments
plot(causal_graph,
     causal_strengths=convert_to_percentage(arrow_strengths),
     figure_size=[15, 10])


# or save the graph
pyd.write_png('Django_arrow_strength.png')

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)


estimate= model.estimate_effect(
 identified_estimand,
 method_name='backdoor.linear_regression',
 confidence_intervals=True,
  test_significance=True
)

print(f'Estimate of causal effect: {estimate}')

Refutation
"""
# Add a random common cause
refutel_common_cause=model.refute_estimate(identified_estimand,estimate,"random_common_cause")
print(refutel_common_cause)


# Use a subset of data
refutel_common_cause=model.refute_estimate(identified_estimand,estimate,"data_subset_refuter")
print(refutel_common_cause)

# Placebo treatment
refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=100)
print(refutation)


We need to ensure that the dataframe used for the OLS regressions (df in this case) does not contain any NaN or infinite values in the columns being used in the models. We can achieve this by applying imputation or dropping rows with missing values specifically before performing these OLS calculations. Given that df was not explicitly imputed in the code leading up to this error, we should apply imputation to the relevant columns of df.
"""

# Step 1: Estimate effect of X on M
# Impute missing values in the relevant columns of df before using them in OLS
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Apply imputation to 'nf' and 'nuc' columns in the main df
df[['nf', 'nuc']] = imputer.fit_transform(df[['nf', 'nuc']])

# Replace inf with large finite values in the relevant columns of df
df[['nf', 'nuc']] = df[['nf', 'nuc']].replace([np.inf, -np.inf], np.nan)
df[['nf', 'nuc']] = imputer.fit_transform(df[['nf', 'nuc']]) # Impute again after replacing inf


model_XM = sm.OLS(df["nuc"], sm.add_constant(df["nf"])).fit()
df["M_hat"] = model_XM.predict(sm.add_constant(df["nf"]))

# Impute missing values in the relevant columns of df before using them in the second OLS model
# Apply imputation to 'days_to_first_fix' and 'M_hat' columns in the main df
df[['days_to_first_fix', 'M_hat']] = imputer.fit_transform(df[['days_to_first_fix', 'M_hat']])

# Replace inf with large finite values in the relevant columns of df
df[['days_to_first_fix', 'M_hat']] = df[['days_to_first_fix', 'M_hat']].replace([np.inf, -np.inf], np.nan)
df[['days_to_first_fix', 'M_hat']] = imputer.fit_transform(df[['days_to_first_fix', 'M_hat']]) # Impute again after replacing inf


# Step 2: Estimate effect of M_hat on Y
model_MY = sm.OLS(df["days_to_first_fix"], sm.add_constant(df["M_hat"])).fit()
print("Estimated causal effect:", model_MY.params["M_hat"])

