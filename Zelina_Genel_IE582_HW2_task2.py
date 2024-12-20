#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:11:13 2024

@author: zelina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

match_data = pd.read_csv("/Users/zelina/Desktop/IE582/HW2/match_data.csv")

first_half_df_2 = match_data[match_data["halftime"] == "1st-half"]
second_half_df_2 = match_data[match_data["halftime"] == "2nd-half"]

# Identify games with a red card in the first 15 minutes
red_card_games = first_half_df_2[(first_half_df_2["minute"] <= 20) & ((first_half_df_2["Redcards - away"] == 1) | (first_half_df_2["Redcards - home"] == 1))]["fixture_id"].unique()

# Remove these games from the dataset
first_half_df_2_remove_red = first_half_df_2[~first_half_df_2["fixture_id"].isin(red_card_games)].reset_index(drop=True)

num_removed_games_red = len(red_card_games)

# probabilities based on odd for 1st half and 2nd half after removal of red card and goal games

first_half_df_2_remove_red["p_home_win"] = 1/first_half_df_2_remove_red["1"]
first_half_df_2_remove_red["p_away_win"] = 1/first_half_df_2_remove_red["2"]
first_half_df_2_remove_red["p_tie"] = 1/first_half_df_2_remove_red["X"]



# Calculate the difference between P(home win) and P(away win) for 1st half
first_half_df_2_remove_red["p_home_minus_away"] = first_half_df_2_remove_red["p_home_win"] - first_half_df_2_remove_red["p_away_win"]

# Define bins from -1 to 1 with a step of 0.05
bins_minus = np.arange(-1, 1.2, 0.2)  # Include 1.0 in the range
bin_labels_minus = [f"({bins_minus[i]:.2f}, {bins_minus[i+1]:.2f}]" for i in range(len(bins_minus) - 1)]

# Categorize the probabilities into bins
first_half_df_2_remove_red["bin_minus"] = pd.cut(first_half_df_2_remove_red["p_home_minus_away"], bins=bins_minus, labels=bin_labels_minus, include_lowest=True)

# Group by bins and result to count outcomes in each bin
bin_result_counts_minus_remove_red = first_half_df_2_remove_red.groupby(["bin_minus", "result"]).size().unstack(fill_value=0)

# Add totals for each bin and normalize counts by bin total
bin_result_counts_minus_remove_red["total"] = bin_result_counts_minus_remove_red.sum(axis=1)
bin_result_counts_minus_remove_red["draw_fraction"] = bin_result_counts_minus_remove_red["X"] / bin_result_counts_minus_remove_red["total"]
#bin_result_counts_minus["home_win_fraction"] = bin_result_counts_minus["1"] / bin_result_counts_minus["total"]
#bin_result_counts_minus["away_win_fraction"] = bin_result_counts_minus["2"] / bin_result_counts_minus["total"]

# View the result
print(bin_result_counts_minus_remove_red)

overlay_x3 = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
overlay_y3 = [0.075881, 0.171190, 0.266027, 0.274446, 0.278935, 0.361219, 0.350442, 0.270294, 0.141387, 0.089564]

# Plot P(home win) - P(away win) on x-axis and P(tie) on y-axis for 1st half
plt.figure(figsize=(6, 4))
plt.scatter(first_half_df_2_remove_red["p_home_minus_away"], first_half_df_2_remove_red["p_tie"], alpha=0.7, color='blue', edgecolor='k')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  

plt.scatter(overlay_x3, overlay_y3, alpha=1.0, color='red', edgecolor='black', s=100, label='Real Probabilities')

plt.xlabel("P(home win) - P(away win)")
plt.ylabel("P(tie)")
plt.title("1st Half-removed: P(home win) - P(away win) vs. P(tie)")

plt.grid(alpha=0.3)
plt.show()

# Identify games with a goal in the last ~5 minutes
goal_games = second_half_df_2[(second_half_df_2["minute"] >= 40) & ((second_half_df_2["Score Change - away"] == 1) 
                                                                    | (second_half_df_2["Score Change - home"] == 1))]["fixture_id"].unique()

# Remove these games from the dataset
second_half_df_2_remove_goal = second_half_df_2[~second_half_df_2["fixture_id"].isin(goal_games)].reset_index(drop=True)

num_removed_games_goal = len(goal_games)
# Display the filtered dataset
# print(second_half_df_2_remove_goal)
second_half_df_2_remove_goal["p_home_win"] = 1/second_half_df_2_remove_goal["1"]
second_half_df_2_remove_goal["p_away_win"] = 1/second_half_df_2_remove_goal["2"]
second_half_df_2_remove_goal["p_tie"] = 1/second_half_df_2_remove_goal["X"]


# Calculate the difference between P(home win) and P(away win) for 2nd half
second_half_df_2_remove_goal["p_home_minus_away"] = second_half_df_2_remove_goal["p_home_win"] - second_half_df_2_remove_goal["p_away_win"]


second_half_df_2_remove_goal["bin_minus"] = pd.cut(second_half_df_2_remove_goal["p_home_minus_away"], bins=bins_minus, labels=bin_labels_minus, include_lowest=True)

# Group by bins and result to count outcomes in each bin
bin_result_counts_minus_remove_goal = second_half_df_2_remove_goal.groupby(["bin_minus", "result"]).size().unstack(fill_value=0)

# Add totals for each bin and normalize counts by bin total
bin_result_counts_minus_remove_goal["total"] = bin_result_counts_minus_remove_goal.sum(axis=1)
bin_result_counts_minus_remove_goal["draw_fraction"] = bin_result_counts_minus_remove_goal["X"] / bin_result_counts_minus_remove_goal["total"]

print(bin_result_counts_minus_remove_goal)


overlay_x4 = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
overlay_y4 = [0.011953, 0.149219, 0.337192, 0.530070, 0.646821, 0.740000, 0.596052, 0.284173, 0.138439, 0.026003]
# Plot P(home win) - P(away win) on x-axis and P(tie) on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(second_half_df_2_remove_goal["p_home_minus_away"], second_half_df_2_remove_goal["p_tie"], alpha=0.7, color='blue', edgecolor='k')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  

plt.scatter(overlay_x4, overlay_y4, alpha=1.0, color='red', edgecolor='black', s=100, label='Real Probabilities')

plt.xlabel("P(home win) - P(away win)")
plt.ylabel("P(tie)")
plt.title("2nd Half-removed: P(home win) - P(away win) vs. P(tie)")

plt.grid(alpha=0.3)
plt.show()








