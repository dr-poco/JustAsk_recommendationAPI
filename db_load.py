import faiss
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer

# --- Tag Pruning Functions ---
def analyze_tags_by_category(df, category_column, tags_column):
    """
    Analyze tags by category to identify common and differentiating tags.
    
    Returns:
    - dict: Dictionary with category as key and list of common tags as value
    - dict: Dictionary with offer_id as key and list of differentiating tags as value
    """
    # Group by category
    category_groups = df.groupby(category_column)
    
    # Dictionary to store common tags by category
    common_tags_by_category = {}
    # Dictionary to store differentiating tags by offer
    differentiating_tags_by_offer = {}
    
    for category, group in category_groups:
        # Extract all tags in this category
        all_tags = []
        for tags_str in group[tags_column]:
            if isinstance(tags_str, str) and tags_str.strip():
                all_tags.extend(tags_str.split('|'))
        
        # Count tag occurrences
        tag_counts = Counter(all_tags)
        
        # Calculate threshold for common tags (present in more than 50% of offers)
        threshold = len(group) * 0.4
        
        # Identify common tags in this category
        common_tags = [tag for tag, count in tag_counts.items() if count >= threshold]
        common_tags_by_category[category] = common_tags
        
        # For each offer in this category, identify differentiating tags
        for _, row in group.iterrows():
            offer_id = row['cdf_offer_id']
            if isinstance(row[tags_column], str) and row[tags_column].strip():
                offer_tags = row[tags_column].split('|')
                
                # Filter out common tags from offer tags before calculating importance
                unique_offer_tags = [tag for tag in offer_tags if tag not in common_tags]
                
                # If no unique tags are left, use all tags but mark them as less important
                if not unique_offer_tags:
                    unique_offer_tags = offer_tags
                
                # Calculate tag importance score (inverse of frequency in category)
                tag_importance = {}
                for tag in unique_offer_tags:
                    if tag in tag_counts:
                        # Inverse frequency - rarer tags get higher scores
                        tag_importance[tag] = 1.0 / (tag_counts[tag] / len(group))
                
                # Sort tags by importance score (highest first)
                sorted_tags = sorted(tag_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Get the top 15 differentiating tags (or all if less than 15)
                top_tags = [tag for tag, _ in sorted_tags[:12]]
                differentiating_tags_by_offer[offer_id] = top_tags
    
    return common_tags_by_category, differentiating_tags_by_offer

# --- 1. Load and Process Data ---
# Load merchant data
merchant_df = pd.read_csv("data/merchants.csv")

# Clean and Prepare Merchant Data
merchant_df_cleaned = merchant_df[['cdf_offer_id', 'cdf_merchant_id', 'offer_name',
                                'merchant_description', 'category', 'curated_image', 'Tags', 'city']].dropna()

# Analyze tags to separate common and differentiating tags
common_tags_by_category, differentiating_tags_by_offer = analyze_tags_by_category(
    merchant_df_cleaned, 'category', 'Tags'
)

# Additional checkpoint: Verify that differentiating tags don't include common tags
print("Verifying that differentiating tags don't include common tags...")
for category, common_tags in common_tags_by_category.items():
    # Get all offers in this category
    category_offers = merchant_df_cleaned[merchant_df_cleaned['category'] == category]['cdf_offer_id'].tolist()
    
    for offer_id in category_offers:
        if offer_id in differentiating_tags_by_offer:
            # Remove any common tags that might have slipped through
            differentiating_tags_by_offer[offer_id] = [
                tag for tag in differentiating_tags_by_offer[offer_id] 
                if tag not in common_tags
            ]
            
            # If we've removed all tags, use the 15 rarest tags for this offer
            if not differentiating_tags_by_offer[offer_id]:
                # Get the offer's tags
                offer_tags_str = merchant_df_cleaned.loc[
                    merchant_df_cleaned['cdf_offer_id'] == offer_id, 'Tags'
                ].iloc[0]
                
                if isinstance(offer_tags_str, str) and offer_tags_str.strip():
                    offer_tags = offer_tags_str.split('|')
                    
                    # Count occurrences of each tag in this category
                    category_tags = []
                    for tags_str in merchant_df_cleaned[merchant_df_cleaned['category'] == category]['Tags']:
                        if isinstance(tags_str, str) and tags_str.strip():
                            category_tags.extend(tags_str.split('|'))
                    
                    tag_counts = Counter(category_tags)
                    
                    # Calculate rarity score for each tag
                    tag_rarity = {tag: 1.0 / tag_counts.get(tag, 1) for tag in offer_tags if tag not in common_tags}
                    
                    # Sort by rarity and take top 15
                    sorted_tags = sorted(tag_rarity.items(), key=lambda x: x[1], reverse=True)
                    differentiating_tags_by_offer[offer_id] = [tag for tag, _ in sorted_tags[:15]]

print("Verification complete!")

# Add new columns to the dataframe
merchant_df_cleaned['differentiating_tags'] = merchant_df_cleaned['cdf_offer_id'].map(
    lambda x: differentiating_tags_by_offer.get(x, [])
)

merchant_df_cleaned['common_tags'] = merchant_df_cleaned['category'].map(
    lambda x: common_tags_by_category.get(x, [])
)

# Convert list columns to string representation for display
merchant_df_cleaned['differentiating_tags_str'] = merchant_df_cleaned['differentiating_tags'].apply(
    lambda x: '|'.join(x) if isinstance(x, list) else ''
)

merchant_df_cleaned['common_tags_str'] = merchant_df_cleaned['common_tags'].apply(
    lambda x: '|'.join(x) if isinstance(x, list) else ''
)

# Save to CSV
merchant_df_cleaned.to_csv("logs/merchant_df_cleaned.csv", index=False)

print("✅ Pruned tags CSV with list format and tag count saved as 'logs/merchant_df_cleaned.csv'.")

# --- 2. Compute Embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create rich text representation for embedding - use differentiating tags instead of all tags
merchant_df_cleaned["text"] = (
    merchant_df_cleaned["offer_name"] + ". " +
    merchant_df_cleaned["merchant_description"] + ". " +
    merchant_df_cleaned["differentiating_tags_str"] + ". " +  # Use differentiating tags
    "Location: " + merchant_df_cleaned["city"] + ". " +
    "Category: " + merchant_df_cleaned["category"]
)

# Your cleaned text data for embedding
text_data = merchant_df_cleaned["text"].tolist()

# Generate embeddings
offer_embeddings = model.encode(text_data, show_progress_bar=True)

# --- 3. Create FAISS Index ---
dimension = offer_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(offer_embeddings).astype("float32"))

# --- 4. Prepare Metadata ---
metadata = []
for _, row in merchant_df_cleaned.iterrows():
    metadata.append({
        "id": str(row["cdf_offer_id"]),
        "user_id": row["cdf_merchant_id"],
        "type": row["category"],
        "title": row["offer_name"],
        "description": row["merchant_description"],
        "image": row["curated_image"],
        "tags": row["differentiating_tags"],  # Use list of differentiating tags
        "city": row["city"],
        "offer_title": row["offer_name"]
    })

# --- 5. Save FAISS Index to File ---
faiss.write_index(faiss_index, "db/faiss_index.bin")

# --- 6. Save Metadata to File ---
with open("db/faiss_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# --- 7. Save Location and Category Information Separately ---
all_locations = set(merchant_df_cleaned["city"].str.lower().unique())
all_categories = set(merchant_df_cleaned["category"].str.lower().unique())

location_category_data = {
    "locations": list(all_locations),
    "categories": list(all_categories)
}

with open("db/location_category_data.pkl", "wb") as f:
    pickle.dump(location_category_data, f)

# Display sample of the processed data
print("\n✅ FAISS index, metadata, and location/category data saved locally.")
print("\n--- Sample of processed data ---")
sample_df = merchant_df_cleaned[['cdf_offer_id', 'offer_name', 'category', 'differentiating_tags_str', 'common_tags_str']].head(3)
print(sample_df)

# Save Tag Distribution Statistics to a text file
with open("logs/tag_distribution_stats.txt", "w") as stats_file:
    stats_file.write("--- Tag Distribution Statistics ---\n")
    for category in common_tags_by_category:
        offers_in_category = len(merchant_df_cleaned[merchant_df_cleaned['category'] == category])
        diff_tag_counts = [len(differentiating_tags_by_offer.get(offer_id, [])) 
                          for offer_id in merchant_df_cleaned[merchant_df_cleaned['category'] == category]['cdf_offer_id']]
        
        stats_file.write(f"\nCategory: {category}\n")
        stats_file.write(f"Number of offers: {offers_in_category}\n")
        stats_file.write(f"Common tags ({len(common_tags_by_category[category])}): {', '.join(common_tags_by_category[category][:5])}...\n" if common_tags_by_category[category] else "No common tags\n")
        stats_file.write(f"Avg differentiating tags per offer: {sum(diff_tag_counts)/len(diff_tag_counts) if diff_tag_counts else 0:.1f}\n")
        stats_file.write(f"Min differentiating tags: {min(diff_tag_counts) if diff_tag_counts else 0}\n")
        stats_file.write(f"Max differentiating tags: {max(diff_tag_counts) if diff_tag_counts else 0}\n")