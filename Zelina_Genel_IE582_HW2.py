#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:20:01 2024

@author: zelina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

match_data = pd.read_csv("/Users/zelina/Desktop/IE582/HW2/match_data.csv")



# filter match data by excluding rows where suspended or stopped value is True
filt_match_data_ss = match_data[~(match_data["suspended"] | match_data["stopped"])]
warnings.simplefilter(action='ignore', category=filt_match_data_ss.errors.SettingWithCopyWarning)
#resetting indexing
#filt_match_data_ss = filt_match_data_ss.reset_index(drop=True)

first_half_df = filt_match_data_ss[filt_match_data_ss["halftime"] == "1st-half"]
second_half_df = filt_match_data_ss[filt_match_data_ss["halftime"] == "2nd-half"]

# check missing values of odds in filtered data
missing_values = first_half_df[["1", "2", "X"]].isna().sum()
print("Missing values 1, X, 2:")
print(missing_values)

missing_values = second_half_df[["1", "2", "X"]].isna().sum()
print("Missing values 1, X, 2:")
print(missing_values)
# there is no missing value of odds

# number of games 
unique_ngame = filt_match_data_ss["fixture_id"].nunique()
print(unique_ngame)

# probabilities based on odd

first_half_df["p_home_win"] = 1/first_half_df["1"]
first_half_df["p_away_win"] = 1/first_half_df["2"]
first_half_df["p_tie"] = 1/first_half_df["X"]

second_half_df["p_home_win"] = 1/second_half_df["1"]
second_half_df["p_away_win"] = 1/second_half_df["2"]
second_half_df["p_tie"] = 1/second_half_df["X"]

#half_prob_avg = filt_match_data_ss.groupby(["fixture_id", "halftime"])[["p_home_win", "p_tie", "p_away_win"]].mean()

# normalized probabilities
first_half_df["p_total"] = first_half_df["p_home_win"] + first_half_df["p_tie"] + first_half_df["p_away_win"]
first_half_df["p_home_win_norm"] = first_half_df["p_home_win"]/first_half_df["p_total"]
first_half_df["p_away_win_norm"] = first_half_df["p_away_win"]/first_half_df["p_total"]
first_half_df["p_tie_norm"] = first_half_df["p_tie"]/first_half_df["p_total"]

second_half_df["p_total"] = second_half_df["p_home_win"] + second_half_df["p_tie"] + second_half_df["p_away_win"]
second_half_df["p_home_win_norm"] = second_half_df["p_home_win"]/second_half_df["p_total"]
second_half_df["p_away_win_norm"] = second_half_df["p_away_win"]/second_half_df["p_total"]
second_half_df["p_tie_norm"] = second_half_df["p_tie"]/second_half_df["p_total"]

#half_prob_avg_norm = filt_match_data_ss.groupby(["fixture_id", "halftime"])[["p_home_win_norm", "p_tie_norm", "p_away_win_norm"]].mean()



# Define bins from 0 to 1 with a step of 0.05
bins = np.arange(0, 1.05, 0.05)  # Include 1.0 in the range
bin_labels = [f"({bins[i]:.2f}, {bins[i+1]:.2f}]" for i in range(len(bins) - 1)]

# categorize the probabilities into bins
first_half_df["bin"] = pd.cut(first_half_df["p_tie_norm"], bins=bins, labels=bin_labels, include_lowest=True)

# group by bins and result to count outcomes in each bin
bin_result_counts = first_half_df.groupby(["bin", "result"]).size().unstack(fill_value=0)

# Add totals for each bin and normalize counts by bin total
bin_result_counts["total"] = bin_result_counts.sum(axis=1)
bin_result_counts["draw_fraction"] = bin_result_counts["X"] / bin_result_counts["total"]
bin_result_counts["home_win_fraction"] = bin_result_counts["1"] / bin_result_counts["total"]
bin_result_counts["away_win_fraction"] = bin_result_counts["2"] / bin_result_counts["total"]

print(bin_result_counts)

#  categorize the probabilities into bins
second_half_df["bin"] = pd.cut(second_half_df["p_tie_norm"], bins=bins, labels=bin_labels, include_lowest=True)

# group by bins and result to count outcomes in each bin
bin_result_counts_2 = second_half_df.groupby(["bin", "result"]).size().unstack(fill_value=0)

# Add totals for each bin and normalize counts by bin total
bin_result_counts_2["total"] = bin_result_counts_2.sum(axis=1)
bin_result_counts_2["draw_fraction"] = bin_result_counts_2["X"] / bin_result_counts_2["total"]
bin_result_counts_2["home_win_fraction"] = bin_result_counts_2["1"] / bin_result_counts_2["total"]
bin_result_counts_2["away_win_fraction"] = bin_result_counts_2["2"] / bin_result_counts_2["total"]


print(bin_result_counts_2)


# Calculate the difference between P(home win) and P(away win) for 1st half
first_half_df["p_home_minus_away"] = first_half_df["p_home_win"] - first_half_df["p_away_win"]

# Define bins from -1 to 1 with a step of 0.05
bins_minus = np.arange(-1, 1.2, 0.2)  # Include 1.0 in the range
bin_labels_minus = [f"({bins_minus[i]:.2f}, {bins_minus[i+1]:.2f}]" for i in range(len(bins_minus) - 1)]

# Categorize the probabilities into bins
first_half_df["bin_minus"] = pd.cut(first_half_df["p_home_minus_away"], bins=bins_minus, labels=bin_labels_minus, include_lowest=True)

# Group by bins and result to count outcomes in each bin
bin_result_counts_minus = first_half_df.groupby(["bin_minus", "result"]).size().unstack(fill_value=0)

# Add totals for each bin and normalize counts by bin total
bin_result_counts_minus["total"] = bin_result_counts_minus.sum(axis=1)
bin_result_counts_minus["draw_fraction"] = bin_result_counts_minus["X"] / bin_result_counts_minus["total"]
#bin_result_counts_minus["home_win_fraction"] = bin_result_counts_minus["1"] / bin_result_counts_minus["total"]
#bin_result_counts_minus["away_win_fraction"] = bin_result_counts_minus["2"] / bin_result_counts_minus["total"]


print(bin_result_counts_minus)

# Data points to overlay
overlay_x = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
overlay_y = [0.082907, 0.165803, 0.272727, 0.274830, 0.281860, 0.360765, 0.353692, 0.270369, 0.141440, 0.088492]

# Plot P(home win) - P(away win) on x-axis and P(tie) on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(first_half_df["p_home_minus_away"], first_half_df["p_tie"], alpha=0.7, color='blue', edgecolor='k')

# Overlay additional data points
plt.scatter(overlay_x, overlay_y, alpha=1.0, color='red', edgecolor='black', s=100, label='Real Probabilities')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  


plt.xlabel("P(home win) - P(away win)")
plt.ylabel("P(tie)")
plt.title("1st Half: P(home win) - P(away win) vs. P(tie)")


plt.grid(alpha=0.3)
plt.show()





# Calculate the difference between P(home win) and P(away win) for 2nd half
second_half_df["p_home_minus_away"] = second_half_df["p_home_win"] - second_half_df["p_away_win"]

# Plot P(home win) - P(away win) on x-axis and P(tie) on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(second_half_df["p_home_minus_away"], second_half_df["p_tie"], alpha=0.7, color='blue', edgecolor='k')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  


plt.xlabel("P(home win) - P(away win)")
plt.ylabel("P(tie)")
plt.title("2nd Half: P(home win) - P(away win) vs. P(tie)")


plt.grid(alpha=0.3)
plt.show()

# Define bins from -1 to 1 with a step of 0.05 for second half
#bins_minus = np.arange(-1, 1.2, 0.2)  # Include 1.0 in the range
#bin_labels_minus = [f"({bins_minus[i]:.2f}, {bins_minus[i+1]:.2f}]" for i in range(len(bins_minus) - 1)]

# categorize the probabilities into bins
second_half_df["bin_minus"] = pd.cut(second_half_df["p_home_minus_away"], bins=bins_minus, labels=bin_labels_minus, include_lowest=True)

# group by bins and result to count outcomes in each bin
bin_result_counts_minus_2 = second_half_df.groupby(["bin_minus", "result"]).size().unstack(fill_value=0)

# Add totals for each bin and normalize counts by bin total
bin_result_counts_minus_2["total"] = bin_result_counts_minus_2.sum(axis=1)
bin_result_counts_minus_2["draw_fraction"] = bin_result_counts_minus_2["X"] / bin_result_counts_minus_2["total"]

print(bin_result_counts_minus_2["draw_fraction"])



########
# Calculate the difference between P(home win) and P(away win) for 2nd half
second_half_df["p_home_minus_away"] = second_half_df["p_home_win"] - second_half_df["p_away_win"]

# Data points to overlay
overlay_x2 = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
overlay_y2 = [0.103505, 0.230263, 0.386364, 0.453321, 0.543131, 0.601999, 0.414867, 0.223309, 0.243024, 0.083160]

# Plot P(home win) - P(away win) on x-axis and P(tie) on y-axis
plt.figure(figsize=(10, 6))
plt.scatter(second_half_df["p_home_minus_away"], second_half_df["p_tie"], alpha=0.7, color='blue', edgecolor='k')

# Overlay additional data points
plt.scatter(overlay_x2, overlay_y2, alpha=1.0, color='red', edgecolor='black', s=100, label='Real Probabilities')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  


# Add labels and title
plt.xlabel("P(home win) - P(away win)")
plt.ylabel("P(tie)")
plt.title("2nd Half: P(home win) - P(away win) vs. P(tie)")

# Show the plot
plt.grid(alpha=0.3)
plt.show()









