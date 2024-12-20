#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:07:15 2024

@author: zelina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


match_data = pd.read_csv("/Users/zelina/Desktop/IE582/HW2/match_data.csv")
first_half_df_3 = match_data[match_data["halftime"] == "1st-half"]


X = first_half_df_3[["Ball Possession % - home", "Ball Possession % - away", "Dangerous Attacks - home", "Dangerous Attacks - away",
                "Goal Attempts - home", "Goal Attempts - away", "Goals - home", "Goals - away", 
                "Penalties - home", "Penalties - away", "Redcards - home", "Redcards - away", 
                "Score Change - home", "Score Change - away", 
                "Successful Passes Percentage - home", "Successful Passes Percentage - away", 
                ]]  
y = first_half_df_3["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# Initialize the model
tree = DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the model
tree.fit(X_train, y_train)

# Visualize the tree
#plt.figure(figsize=(15, 10))
#plot_tree(tree, feature_names=X.columns, class_names=["1", "X", "2"], filled=True)
#plt.show()

plt.figure(figsize=(30, 20))
plot_tree(tree, feature_names=X.columns, class_names=["1", "X", "2"], filled=True, fontsize=12)
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")  
plt.close()


second_half_df_3 = match_data[match_data["halftime"] == "2nd-half"]
X_2 = second_half_df_3[["Ball Possession % - home", "Ball Possession % - away", "Dangerous Attacks - home", "Dangerous Attacks - away",
                "Goal Attempts - home", "Goal Attempts - away", "Goals - home", "Goals - away", 
                "Penalties - home", "Penalties - away", "Redcards - home", "Redcards - away", 
                "Score Change - home", "Score Change - away", 
                "Successful Passes Percentage - home", "Successful Passes Percentage - away", 
                ]]  
y_2 = second_half_df_3["result"]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3, random_state=42)




# Initialize the model
tree_2 = DecisionTreeClassifier(max_depth=4, random_state=42)  

# Train the model
tree_2.fit(X_train_2, y_train_2)

# Visualize the tree
#plt.figure(figsize=(15, 10))
#plot_tree(tree_2, feature_names=X_2.columns, class_names=["1", "X", "2"], filled=True)
#plt.show()

plt.figure(figsize=(30, 20))
plot_tree(tree_2, feature_names=X_2.columns, class_names=["1", "X", "2"], filled=True, fontsize=12)
plt.savefig("decision_tree_2.png", dpi=300, bbox_inches="tight") 
plt.close()






