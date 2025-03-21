import numpy as np
import torch
import os
from datasets import load_dataset
from tqdm import tqdm
from scipy.sparse import csr_matrix

# Define output directory
output_dir = "./data/amazon/"
os.makedirs(output_dir, exist_ok=True)

# Log file
log_file = os.path.join(output_dir, "log.txt")

# Define categories
categories = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Video_Games", "Unknown"
]

# Initialize log file
with open(log_file, "w") as f:
    f.write("Amazon Review Dataset Statistics\n\n")

# Process each category separately
for category in tqdm(categories, desc="Processing Categories"):
    category_output_dir = os.path.join(output_dir, category)
    os.makedirs(category_output_dir, exist_ok=True)

    # Load dataset for this category
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"0core_rating_only_{category}", trust_remote_code=True)
    data = dataset["full"]

    # Build item-user dictionary
    user_map = {}
    item_map = {}
    ratings = []

    for entry in tqdm(data, desc=f"Processing {category}"):
        user_id = entry['user_id']
        item_id = entry['parent_asin']
        
        # Convert rating to float and ensure it's an integer
        rating = float(entry['rating'])
        if rating % 1 == 0:
            rating = int(rating)  # Keep as integer if whole number
        else:
            print("warning: rating is not a whole number")
            rating = int(round(rating))  # Round to nearest integer if decimal

        # Assign unique indices to users and items
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if item_id not in item_map:
            item_map[item_id] = len(item_map)

        ratings.append((item_map[item_id], user_map[user_id], rating))  # Swap order

    # Convert to sparse matrix
    num_users = len(user_map)
    num_items = len(item_map)

    row_indices = []  # Now item indices
    col_indices = []  # Now user indices
    values = []

    for item_idx, user_idx, rating in ratings:
        row_indices.append(item_idx)  # Items are rows
        col_indices.append(user_idx)  # Users are columns
        values.append(rating)

    # Create a sparse CSR matrix with shape (n_items, n_users)
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_items, num_users), dtype=np.int8)

    # Compute statistics
    num_entries = len(values)
    sparsity_ratio = 1 - (num_entries / (num_items * num_users))

    # Log statistics
    stats = (
        f"Category: {category}\n"
        f"Rows (Items): {num_items}\n"
        f"Columns (Users): {num_users}\n"
        f"Entries (Ratings): {num_entries}\n"
        f"Sparsity Ratio: {sparsity_ratio:.6f}\n\n"
    )

    with open(log_file, "a") as f:
        f.write(stats)

    # Save sparse matrices only
    np.savez_compressed(
        os.path.join(category_output_dir, "rating_matrix_sparse.npz"),
        data=sparse_matrix.data.astype(np.int8),  # Save as int8
        indices=sparse_matrix.indices,
        indptr=sparse_matrix.indptr,
        shape=sparse_matrix.shape
    )
    torch.save(sparse_matrix, os.path.join(category_output_dir, "rating_matrix_sparse.pt"))

    print(f"Saved sparse rating matrices for {category} at {category_output_dir}")

print(f"\nAll category data processed. Log saved at: {log_file}")
