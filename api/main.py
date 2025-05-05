from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import faiss
import pickle
import re
import json
import csv
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Create validation directory if it doesn't exist
os.makedirs("validation", exist_ok=True)

def validate_tag_matching(
    merchant_tags: List[str],
    user_preferred_tags: List[str],
    query_category: str,
    user_name: str,
    query: str,
    offer_id: str = None,
    offer_title: str = None
) -> Dict[str, Any]:
    """
    Validate and score the matching between merchant tags and user preferred tags for a specific offer.
    Returns a dictionary with validation metrics and matching details.
    """
    # Convert all tags to lowercase for comparison
    merchant_tags = [tag.lower() for tag in merchant_tags]
    user_preferred_tags = [tag.lower() for tag in user_preferred_tags]
    
    # Initialize results dictionary
    results = {
        "user_name": user_name,
        "query": query,
        "query_category": query_category,
        "timestamp": datetime.now().isoformat(),
        "offer_id": offer_id,
        "offer_title": offer_title,
        "total_merchant_tags": len(merchant_tags),
        "total_user_tags": len(user_preferred_tags),
        "exact_matches": [],
        "partial_matches": [],
        "unmatched_user_tags": [],
        "unmatched_merchant_tags": [],
        "matching_metrics": {
            "exact_match_count": 0,
            "partial_match_count": 0,
            "exact_match_percentage": 0.0,
            "partial_match_percentage": 0.0,
            "overall_match_score": 0.0
        },
        "match_explanation": ""
    }
    
    # Check for exact matches
    for user_tag in user_preferred_tags:
        if user_tag in merchant_tags:
            results["exact_matches"].append({
                "user_tag": user_tag,
                "merchant_tag": user_tag,
                "similarity_score": 1.0
            })
            results["matching_metrics"]["exact_match_count"] += 1
        else:
            results["unmatched_user_tags"].append(user_tag)
    
    # Check for partial matches using sequence matching
    for user_tag in user_preferred_tags:
        if user_tag not in [m["user_tag"] for m in results["exact_matches"]]:
            best_match = None
            best_score = 0.0
            
            for merchant_tag in merchant_tags:
                if merchant_tag not in [m["merchant_tag"] for m in results["exact_matches"]]:
                    similarity = SequenceMatcher(None, user_tag, merchant_tag).ratio()
                    if similarity > 0.7 and similarity > best_score:  # Threshold for partial match
                        best_match = merchant_tag
                        best_score = similarity
            
            if best_match:
                results["partial_matches"].append({
                    "user_tag": user_tag,
                    "merchant_tag": best_match,
                    "similarity_score": best_score
                })
                results["matching_metrics"]["partial_match_count"] += 1
    
    # Calculate metrics
    total_matches = results["matching_metrics"]["exact_match_count"] + results["matching_metrics"]["partial_match_count"]
    total_tags = len(user_preferred_tags)
    
    if total_tags > 0:
        results["matching_metrics"]["exact_match_percentage"] = (
            results["matching_metrics"]["exact_match_count"] / total_tags * 100
        )
        results["matching_metrics"]["partial_match_percentage"] = (
            results["matching_metrics"]["partial_match_count"] / total_tags * 100
        )
        results["matching_metrics"]["overall_match_score"] = (
            (results["matching_metrics"]["exact_match_count"] * 1.0 +
             results["matching_metrics"]["partial_match_count"] * 0.7) / total_tags
        )
    
    # Add unmatched merchant tags
    matched_merchant_tags = set(
        [m["merchant_tag"] for m in results["exact_matches"]] +
        [m["merchant_tag"] for m in results["partial_matches"]]
    )
    results["unmatched_merchant_tags"] = [
        tag for tag in merchant_tags if tag not in matched_merchant_tags
    ]
    
    # Generate match explanation
    explanation_parts = []
    
    if results["exact_matches"]:
        exact_matches_str = ", ".join([f"'{m['user_tag']}'" for m in results["exact_matches"]])
        explanation_parts.append(f"Exact tag matches: {exact_matches_str}")
    
    if results["partial_matches"]:
        partial_matches_str = ", ".join([
            f"'{m['user_tag']}' (similar to '{m['merchant_tag']}' with {m['similarity_score']:.2f} similarity)"
            for m in results["partial_matches"]
        ])
        explanation_parts.append(f"Partial tag matches: {partial_matches_str}")
    
    if explanation_parts:
        results["match_explanation"] = "This offer was recommended because of: " + "; ".join(explanation_parts)
    else:
        results["match_explanation"] = "This offer was recommended based on general preferences as no specific tag matches were found."
    
    return results

def save_validation_results(results: Dict[str, Any], format: str = "json") -> str:
    """
    Save validation results to a file in the specified format.
    Returns the path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation/{timestamp}_{results['user_name']}_{results['query'].replace(' ', '_')}_{results['query_category']}"
    
    if format.lower() == "json":
        filepath = f"{filename}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
    else:  # CSV format
        filepath = f"{filename}.csv"
        # Flatten the results for CSV
        flat_results = {
            "user_name": results["user_name"],
            "query": results["query"],
            "query_category": results["query_category"],
            "timestamp": results["timestamp"],
            "offer_id": results["offer_id"],
            "offer_title": results["offer_title"],
            "total_merchant_tags": results["total_merchant_tags"],
            "total_user_tags": results["total_user_tags"],
            "exact_match_count": results["matching_metrics"]["exact_match_count"],
            "partial_match_count": results["matching_metrics"]["partial_match_count"],
            "exact_match_percentage": results["matching_metrics"]["exact_match_percentage"],
            "partial_match_percentage": results["matching_metrics"]["partial_match_percentage"],
            "overall_match_score": results["matching_metrics"]["overall_match_score"],
            "exact_matches": "; ".join([f"{m['user_tag']}={m['merchant_tag']}" for m in results["exact_matches"]]),
            "partial_matches": "; ".join([f"{m['user_tag']}={m['merchant_tag']}({m['similarity_score']:.2f})" for m in results["partial_matches"]]),
            "unmatched_user_tags": "; ".join(results["unmatched_user_tags"]),
            "unmatched_merchant_tags": "; ".join(results["unmatched_merchant_tags"]),
            "match_explanation": results["match_explanation"]
        }
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_results.keys())
            writer.writeheader()
            writer.writerow(flat_results)
    
    return filepath

def save_all_validation_results(validation_results: List[Dict[str, Any]], user_name: str, query: str, query_category: str) -> str:
    """
    Save all validation results in a single JSON file.
    Returns the path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation/{timestamp}_{user_name}_{query.replace(' ', '_')}_{query_category}_all_offers.json"
    
    # Create a summary object containing all validation results
    summary = {
        "user_name": user_name,
        "query": query,
        "query_category": query_category,
        "timestamp": datetime.now().isoformat(),
        "total_offers": len(validation_results),
        "offers": validation_results
    }
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
    
    return filename

app = FastAPI()

origins = [
    "https://cd-search-demo.vercel.app",
    "https://justask.tngrm.ai/",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Data on Startup ---
print("Loading user profile data...")  # Debug print
user_profile_df = pd.read_csv("data/user_profile.csv", na_filter=False)  # Don't convert empty strings to NaN
print(f"User profile columns: {user_profile_df.columns.tolist()}")  # Debug print
print(f"Sample user data:\n{user_profile_df.head()}")  # Debug print

# Convert empty strings to None for preferred tags columns
preferred_tags_columns = [col for col in user_profile_df.columns if col.startswith('preferredtags_')]
for col in preferred_tags_columns:
    user_profile_df[col] = user_profile_df[col].replace('', None)
    print(f"Column {col} values:\n{user_profile_df[col].head()}")  # Debug print

model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("db/faiss_index.bin")

with open("db/faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Try to load the location and category data from db_load.py if available
try:
    with open("db/location_category_data.pkl", "rb") as f:
        location_category_data = pickle.load(f)
        loaded_locations = set(location_category_data.get("locations", []))
        loaded_categories = set(location_category_data.get("categories", []))
except (FileNotFoundError, KeyError):
    loaded_locations = set()
    loaded_categories = set()

# --- Known locations and categories ---
# Extract locations and categories from the metadata
all_locations = loaded_locations.copy() if loaded_locations else set()
all_categories = loaded_categories.copy() if loaded_categories else set()

# Add any missing locations and categories from metadata
for item in metadata:
    if "city" in item and item["city"]:
        all_locations.add(item["city"].lower())
    if "type" in item and item["type"]:
        all_categories.add(item["type"].lower())
    if "tags" in item and item["tags"]:
        # Extract tags and add them as potential categories
        tags = item["tags"].lower().split("|") if isinstance(item["tags"], str) else []
        all_categories.update(tags)

# Add variations and common names for locations
location_variations = {
    "dubai": ["dubai", "dxb", "Dubai"],
    "abu dhabi": ["abu dhabi", "ad", "abudhabi", "Abu Dhabi"],
    "sharjah": ["sharjah", "shj", "Sharjah"],
    "ajman": ["ajman", "aj", "Ajman"],
    "ras al khaimah": ["ras al khaimah", "rak"],
    "fujairah": ["fujairah", "fu", "Fujairah"],
    "umm al quwain": ["umm al quwain", "uaq"],
    "al ain": ["al ain", "alain", "Al Ain"],
    "al ain city": ["al ain city", "alain city"],
    "al ain oasis": ["al ain oasis", "alain oasis"],    
    "Ras Al Khaimah": ["Ras Al Khaimah", "RAK"],
    "Umm Al Quwain": ["Umm Al Quwain", "UAQ"],
    "all​": ["all", "everywhere", "anywhere"],
    "all cities": ["all cities", "every city", "any city"],
    "all emirates": ["all emirates", "every emirate", "any emirate"],  
    "all locations": ["all locations", "every location", "any location"],
    "all areas": ["all areas", "every area", "any area"],   
    "all places": ["all places", "every place", "any place"],
    "all regions": ["all regions", "every region", "any region"],
    "all destinations": ["all destinations", "every destination", "any destination"],
    # Add more as needed
}

# Add more locations to the set
for main_loc, variations in location_variations.items():
    for var in variations:
        all_locations.add(var)

# Category and tag mapping for contextual queries
contextual_mappings = {
    # Food related
    "hungry": ["restaurant", "food", "dining", "café", "cafe", "bakery", "lunch", "dinner", "breakfast"],
    "eat": ["restaurant", "food", "dining", "café", "cafe", "bakery", "lunch", "dinner", "breakfast"],
    "food": ["restaurant", "food", "dining", "café", "cafe", "bakery", "lunch", "dinner", "breakfast"],
    "restaurant": ["restaurant", "food", "dining", "café", "cafe", "bakery"],
    "bakery": ["bakery", "bread", "dessert", "pastry"],
    "coffee": ["café", "cafe", "coffee shop", "espresso"],
    
    # Accommodation
    "stay": ["hotel", "resort", "accommodation", "lodging"],
    "sleep": ["hotel", "resort", "accommodation", "lodging"],
    "hotel": ["hotel", "resort", "accommodation", "lodging"],
    "resort": ["resort", "hotel", "spa", "vacation"],
    
    # Shopping
    # "shopping": ["apparel", "mall", "store", "fashion", "boutique"],
    # "buy": ["mall", "store", "boutique", "fashion", "apparel"],
    # "purchase": ["apparel", "mall", "store", "fashion", "boutique"],
    
    # Home items
    "furniture": ["furniture", "home decor", "interior design", "home", "household"],
    "decor": ["home decor", "interior design", "furniture", "household"],
    "home": ["furniture", "home decor", "interior design", "household"],
    
    # Electronics
    "electronics": ["electronics", "gadgets", "tech", "technology", "devices"],
    "gadgets": ["electronics", "gadgets", "tech", "technology", "devices"],
    "tech": ["electronics", "gadgets", "tech", "technology", "devices"],
    
    # Fashion
    "clothing": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothes": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "fashion": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing store": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing shop": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing boutique": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing outlet": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing brand": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing line": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing collection": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing retailer": ["clothing", "fashion", "apparel", "clothes", "wear"],
    "clothing manufacturer": ["clothing", "fashion", "apparel", "clothes", "wear"],
    
    # Travel
    "travel": ["travel", "tourism", "vacation", "tour", "trip"],
    "tour": ["travel", "tourism", "vacation", "tour", "trip"],
    "trip": ["travel", "tourism", "vacation", "tour", "trip"],
    "vacation": ["travel", "tourism", "vacation", "tour", "trip", "holiday"],
    
    # Entertainment
    "entertainment": ["entertainment", "movies", "theater", "cinema", "fun", "leisure"],
    "movies": ["cinema", "theater", "films", "entertainment"],
    "fun": ["entertainment", "leisure", "recreation", "amusement"],
    
    # Health & Wellness
    "health": ["health", "fitness", "wellness", "spa", "gym"],
    "fitness": ["fitness", "gym", "workout", "health", "wellness"],
    "wellness": ["wellness", "spa", "health", "fitness", "beauty"],
    
    # Beauty
    "beauty": ["beauty", "salon", "cosmetics", "makeup", "skincare"],
    "salon": ["salon", "beauty", "haircare", "spa"],
    
    # Sports
    "sports": ["sports", "fitness", "athletic", "active", "recreation"],
    "active": ["sports", "fitness", "athletic", "active", "recreation"],
    
    # Add more mappings as needed
}

# --- Category Mapping ---
category_to_profile_column = {
    "food": "preferredtags_dining",
    "dining": "preferredtags_dining",
    "restaurant": "preferredtags_dining",
    "apparel": "preferredtags_apparel",
    "clothing": "preferredtags_apparel",
    "fashion": "preferredtags_apparel",
    "retail": "preferredtags_retail",
    "shopping": "preferredtags_retail",
    "store": "preferredtags_retail",
    "travel": "preferredtags_travel",
    "tourism": "preferredtags_travel",
    "personal": "preferredtags_personalserviceproviders",
    "service": "preferredtags_personalserviceproviders",
    "entertainment": "preferredtags_entertainment",
    "leisure": "preferredtags_entertainment",
    "fun": "preferredtags_entertainment"
}

# Mapping from query context to category
contextual_to_profile_column = {
    "hungry": "preferredtags_dining",
    "eat": "preferredtags_dining",
    "food": "preferredtags_dining",
    "restaurant": "preferredtags_dining",
    "restaurants": "preferredtags_dining",
    "dining": "preferredtags_dining",
    "clothing": "preferredtags_apparel",
    "clothes": "preferredtags_apparel",
    "fashion": "preferredtags_apparel",
    "apparel": "preferredtags_apparel",
    "wear": "preferredtags_apparel",
    "shopping": "preferredtags_retail",
    "retail": "preferredtags_retail",
    "store": "preferredtags_retail",
    "buy": "preferredtags_retail",
    "purchase": "preferredtags_retail",
    "travel": "preferredtags_travel",
    "tourism": "preferredtags_travel",
    "service": "preferredtags_personalserviceproviders",
    "entertainment": "preferredtags_entertainment"
}

# Category to type mapping for strict filtering
category_to_type = {
    "preferredtags_apparel": ["apparel and accessories"],
    "preferredtags_retail": ["retail stores"],
    "preferredtags_dining": ["dining"],
    "preferredtags_travel": ["travel"],
    "preferredtags_personalserviceproviders": ["personal service providers"],
    "preferredtags_entertainment": ["entertainment"],
    # Add direct mappings for query terms
    "clothing": ["apparel and accessories"],
    "clothes": ["apparel and accessories"],
    "fashion": ["apparel and accessories"],
    "apparel": ["apparel and accessories"],
    "wear": ["apparel and accessories"],
    "shopping": ["retail stores"],
    "retail": ["retail stores"],
    "store": ["retail stores"],
    "buy": ["retail stores"],
    "purchase": ["retail stores"]
}

# --- Pydantic Models ---
class Details(BaseModel):
    priceRange : str
    location: str 
    rating: str
    knownFor: str
    deals: str
    relevance: str
    narrative: str
    offer_title: str
    
class Offer(BaseModel):
    id: str
    type: str
    title: str
    description: str
    image: str
    details: Details
    bookmarked: bool

def extract_constraints_from_query(query, user_name=None):
    """Extract location, category, and tag constraints from the query and user profile"""
    query_lower = query.lower()
    location = None
    categories = set()
    tags = set()
    user_country = None
    user_city = None
    preferred_tags = []
    
    print(f"\nDebug - Starting category mapping for query: '{query_lower}'")  # Debug print
    
    # First check for contextual words that map directly to profile columns
    query_category = None
    
    # Clean the query by removing punctuation and splitting
    clean_words = [word.strip('.,!?') for word in query_lower.split()]
    print(f"Debug - Cleaned words: {clean_words}")  # Debug print
    
    # Check the entire query first for direct mappings
    print(f"Debug - Checking direct mapping for entire query")  # Debug print
    if query_lower in contextual_to_profile_column:
        query_category = contextual_to_profile_column[query_lower]
        print(f"Debug - Found direct mapping for entire query to column: {query_category}")  # Debug print
    else:
        # Then check individual words
        print(f"Debug - Checking individual words in query: {clean_words}")  # Debug print
        for word in clean_words:
            print(f"Debug - Checking word '{word}' in contextual_to_profile_column")  # Debug print
            if word in contextual_to_profile_column:
                query_category = contextual_to_profile_column[word]
                print(f"Debug - Found direct mapping for word '{word}' to column: {query_category}")  # Debug print
                break
    
    # If no direct mapping found, try the contextual category mapping
    if not query_category:
        print(f"Debug - No direct mapping found, trying contextual category mapping")  # Debug print
        # Check the entire query first
        if query_lower in contextual_mappings:
            query_category = contextual_mappings[query_lower]
            print(f"Debug - Found category mapping for entire query to column: {query_category}")  # Debug print
        else:
            # Then check individual words
            for word in clean_words:
                print(f"Debug - Checking word '{word}' in contextual_mappings")  # Debug print
                if word in contextual_mappings:
                    query_category = contextual_mappings[word]
                    print(f"Debug - Found category mapping for word '{word}' to column: {query_category}")  # Debug print
                    break
    
    # If we have dining/restaurant related categories but no direct mapping
    if any(cat in ['restaurant', 'dining', 'cafe', 'food'] for cat in categories):
        query_category = "preferredtags_dining"
    
    print(f"Debug - Final query category: {query_category}")  # Debug print
    
    # If user is provided, get their profile information
    if user_name:
        print(f"\nDebug - Looking up user profile for: {user_name}")  # Debug print
        user_row = user_profile_df[user_profile_df['name'].str.lower() == user_name.lower()]
        if not user_row.empty:
            print(f"Debug - Found user profile for {user_name}")  # Debug print
            
            # Get user's location information
            if 'country' in user_row.columns and not user_row['country'].isna().all():
                user_country = user_row['country'].iloc[0]
                print(f"Debug - User country: {user_country}")  # Debug print
            if 'city' in user_row.columns and not user_row['city'].isna().all():
                user_city = user_row['city'].iloc[0]
                print(f"Debug - User city: {user_city}")  # Debug print
            
            # Get user's preferred tags for the determined category
            if query_category and query_category in user_row.columns:
                preferred_tags_str = user_row[query_category].iloc[0]
                print(f"Debug - Raw preferred tags string for {query_category}: {preferred_tags_str}")  # Debug print
                
                if pd.notna(preferred_tags_str) and isinstance(preferred_tags_str, str):
                    preferred_tags = [tag.strip() for tag in preferred_tags_str.split('|') if tag.strip()]
                    print(f"Debug - Found preferred tags for {user_name}: {preferred_tags}")
                else:
                    print(f"Debug - No preferred tags found for {user_name} in column {query_category}")
            else:
                print(f"Debug - Query category {query_category} not found in user profile columns")
                print(f"Debug - Available columns: {user_row.columns.tolist()}")  # Debug print
        else:
            print(f"Debug - No user profile found for {user_name}")  # Debug print
    
    # Check for locations in the query
    for loc in all_locations:
        pattern = rf'\b{re.escape(loc)}\b'
        if re.search(pattern, query_lower):
            # Normalize to main location name
            for main_loc, variations in location_variations.items():
                if loc.lower() in [var.lower() for var in variations]:
                    location = main_loc
                    break
            if not location:  # If not found in variations
                location = loc
            break
    
    # If no location in query but user has location, use that
    if not location:
        if user_city and user_city.lower() != "all":
            location = user_city
        elif user_country and user_country.lower() != "all":
            location = user_country
    
    # Check for categories in query
    for category in all_categories:
        pattern = rf'\b{re.escape(category)}s?\b'  # Match singular or plural
        if re.search(pattern, query_lower):
            categories.add(category)
    
    # Add user's preferred tags
    if preferred_tags:
        tags.update(preferred_tags)
    
    # Check for contextual clues
    for contextual_word, related_categories in contextual_mappings.items():
        if contextual_word in query_lower:
            categories.update(related_categories)
            tags.update(related_categories)
    
    return location, list(categories), list(tags), query_category, preferred_tags

def check_relevance(item, location_constraint, category_constraints, tag_constraints, preferred_tags):
    """Check if an item matches the given constraints with emphasis on user preferences"""
    relevance_score = 0
    
    # If we have category constraints, first check if the item's type matches
    if category_constraints:
        item_type = item.get("type", "").lower()
        # Get the allowed types for the category
        allowed_types = []
        for cat in category_constraints:
            if cat in category_to_type:
                allowed_types.extend(category_to_type[cat])
            # Also check if the category constraint itself is a valid type
            # Flatten all values in category_to_type and check if cat matches any of them
            all_valid_types = [t.lower() for types in category_to_type.values() for t in (types if isinstance(types, list) else [types])]
            if cat.lower() in all_valid_types:
                allowed_types.append(cat)
        
        # Convert to lowercase for comparison
        allowed_types = [t.lower() for t in allowed_types]
        
        # If the item's type doesn't match any of the allowed types, return False immediately
        if not any(t in item_type for t in allowed_types):
            print(f"Item type '{item_type}' not in allowed types: {allowed_types}")
            return False, 0
    
    # Check location constraint
    location_match = False
    if location_constraint and "city" in item:
        if item["city"] and (
            location_constraint.lower() in item["city"].lower() or 
            item["city"].lower() in ["all", "all​"] or
            location_constraint.lower() in ["all", "all​", "all cities", "all emirates", "all locations"]
        ):
            location_match = True
            relevance_score += 1
    elif location_constraint and "country" in item:
        if item["country"] and (
            location_constraint.lower() in item["country"].lower() or 
            item["country"].lower() in ["all", "all​"] or
            location_constraint.lower() in ["all", "all​", "all countries"]
        ):
            location_match = True
            relevance_score += 1
    else:
        location_match = True  # No location constraint or no location in item
    
    # If location doesn't match at all, return False
    if not location_match and location_constraint:
        if location_constraint.lower() not in ["all", "all​", "all cities", "all emirates", "all locations"]:
            return False, 0
    
    # Check category constraints again for scoring
    category_match = False
    if category_constraints and "type" in item:
        if item["type"] and any(cat.lower() in item["type"].lower() for cat in category_constraints):
            category_match = True
            relevance_score += 2  # Increased weight for category match
        else:
            # If no category match, check if any keywords in the title or description match
            title_match = any(cat.lower() in item["title"].lower() for cat in category_constraints)
            desc_match = any(cat.lower() in item["description"].lower() for cat in category_constraints)
            if title_match or desc_match:
                category_match = True
                relevance_score += 1
    else:
        category_match = True  # No category constraints or no type in item
    
    # Check tag constraints with emphasis on preferred tags
    tag_match = False
    if (tag_constraints or preferred_tags) and "tags" in item:
        if item["tags"]:
            # Extract individual tags
            if isinstance(item["tags"], str):
                item_tags = item["tags"].lower().split("|")
            else:
                item_tags = [tag.lower() for tag in item["tags"]]
            
            # Check for matches with all tag constraints
            if tag_constraints:
                for tag in tag_constraints:
                    if tag.lower() in item_tags:
                        tag_match = True
                        relevance_score += 0.5
            
            # Give higher weight to preferred tags
            if preferred_tags:
                for tag in preferred_tags:
                    if tag.lower() in item_tags:
                        tag_match = True
                        relevance_score += 2  # Higher score for preferred tags
        else:
            tag_match = not (tag_constraints or preferred_tags)  # Match only if no tags are required
    else:
        tag_match = True  # No tag constraints or no tags in item
    
    # If tags don't match at all and we have tag constraints, return False
    if not tag_match and (tag_constraints or preferred_tags):
        return False, 0
    
    return True, relevance_score

# --- API Endpoint ---
@app.get("/")
async def main():
    return {"message": "Hello to the recommendation engine"}

@app.get("/user-metadata")
def get_user_metadata():
    global user_profile_df

    # Ensure the user_profile_df is available
    if user_profile_df.empty:
        return JSONResponse(status_code=404, content={"error": "User profile data not found"})

    country_column = "country"
    if country_column not in user_profile_df.columns:
        return JSONResponse(status_code=400, content={"error": f"'{country_column}' column not found in user profile"})

    # Get unique country names
    countries = sorted(user_profile_df[country_column].dropna().unique().tolist())

    # Replace NaN with None to make JSON serializable
    sanitized_df = user_profile_df.replace({np.nan: None})

    # Convert each row to a dictionary (as a list of user profiles)
    user_profiles = sanitized_df.to_dict(orient="records")

    return {
        "countries": countries,
        "user_profiles": user_profiles
    }

    
@app.get("/recommendations", response_model=List[Offer])
def get_recommendations(
    user_name: str = Query(..., description="User ID"),
    query: Optional[str] = Query(None, description="Custom query text (optional)"),
    top_k: int = Query(20, description="Initial number of results to fetch before filtering")
):
    # If no query is provided, use a default one
    if not query:
        query = "Recommend me something"
    
    # Extract location, category, tag constraints and user preferences
    location_constraint, category_constraints, tag_constraints, query_category, preferred_tags = extract_constraints_from_query(query, user_name)
    
    # Log the constraints for debugging
    print(f"User: {user_name}")
    print(f"Query: {query}")
    print(f"Query Category: {query_category}")
    print(f"Location: {location_constraint}")
    print(f"Categories: {category_constraints}")
    print(f"Tags: {tag_constraints}")
    print(f"Preferred Tags: {preferred_tags}")
    
    # Enhance query with context if we found constraints
    enhanced_query = query
    if location_constraint:
        enhanced_query += f" location:{location_constraint}"
    if category_constraints:
        enhanced_query += f" category:{' '.join(category_constraints)}"
    if tag_constraints:
        enhanced_query += f" tags:{' '.join(tag_constraints)}"
    if preferred_tags:
        enhanced_query += f" preferredTags:{' '.join(preferred_tags)}"
    
    # Fetch more results initially as we'll be filtering them
    query_vector = model.encode([enhanced_query]).astype("float32")
    D, I = faiss_index.search(query_vector, top_k * 5)  # Get more results initially
    
    # Store results with their relevance scores
    results_with_scores = []
    
    # Get allowed types for the category constraints
    allowed_types = []
    if category_constraints:
        for cat in category_constraints:
            if cat in category_to_type:
                allowed_types.extend(category_to_type[cat])
        allowed_types = [t.lower() for t in allowed_types]
        print(f"Allowed types for filtering: {allowed_types}")
    
    # Initialize list to store all validation results
    all_validation_results = []
    
    for idx in I[0]:
        if idx >= len(metadata):  # Safety check
            continue
            
        m = metadata[idx]
        
        # Apply all constraints and get relevance score
        is_relevant, relevance_score = check_relevance(m, location_constraint, category_constraints, tag_constraints, preferred_tags)
        if not is_relevant:
            continue
        
        # Extract merchant tags for validation
        merchant_tags = []
        if "tags" in m and m["tags"]:
            if isinstance(m["tags"], str):
                merchant_tags = m["tags"].lower().split("|")
            else:
                merchant_tags = [tag.lower() for tag in m["tags"]]
        
        # Perform per-offer validation
        validation_results = validate_tag_matching(
            merchant_tags=merchant_tags,
            user_preferred_tags=preferred_tags,
            query_category=query_category,
            user_name=user_name,
            query=query,
            offer_id=m["id"],
            offer_title=m["title"]
        )
        
        # Add validation results to the list
        all_validation_results.append(validation_results)
        
        # Build the result with relevance score and validation explanation
        results_with_scores.append((
            {
                "id": m["id"],
                "type": m["type"],
                "title": m["title"],
                "description": m["description"],
                "image": m["image"],
                "details": {
                    "priceRange": "",
                    "location": m.get("city", "") or m.get("country", ""),
                    "rating": "",
                    "knownFor": "",
                    "deals": "",
                    "relevance": validation_results["match_explanation"],
                    "narrative": f"Recommended based on {' and '.join(category_constraints) if category_constraints else 'general'} preferences",
                    "offer_title": m.get("offer_title", m["title"])
                },
                "bookmarked": False
            },
            relevance_score
        ))
    
    # Sort results by relevance score and take the top_k
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    results = [item[0] for item in results_with_scores[:top_k]]
    
    # Save all validation results in a single file
    validation_file = save_all_validation_results(all_validation_results, user_name, query, query_category)
    print(f"\nAll validation results saved to: {validation_file}")
    
    # Print validation summary for all offers
    print("\nTag Matching Validation Summary for All Offers:")
    for validation in all_validation_results:
        print(f"\nOffer: {validation['offer_title']} (ID: {validation['offer_id']})")
        print(f"Total Merchant Tags: {validation['total_merchant_tags']}")
        print(f"Total User Tags: {validation['total_user_tags']}")
        print(f"Exact Matches: {validation['matching_metrics']['exact_match_count']}")
        print(f"Partial Matches: {validation['matching_metrics']['partial_match_count']}")
        print(f"Exact Match Percentage: {validation['matching_metrics']['exact_match_percentage']:.2f}%")
        print(f"Partial Match Percentage: {validation['matching_metrics']['partial_match_percentage']:.2f}%")
        print(f"Overall Match Score: {validation['matching_metrics']['overall_match_score']:.2f}")
        print(f"Match Explanation: {validation['match_explanation']}")
    
    return results