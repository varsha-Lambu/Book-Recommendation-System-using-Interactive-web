from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import difflib
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
import os
import re

app = Flask(__name__)

# HTML template embedded directly in the Python file
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Book Recommendation System</title>
    <style>
        :root {
            --primary-color: #6d28d9;
            --primary-dark: #5b21b6;
            --secondary-color: #f8f9fa;
            --text-color: #333;
            --light-gray: #e9ecef;
            --border-radius: 10px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f0f2f5;
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: var(--box-shadow);
        }
        
        .container {
            max-width: 900px;
            margin: 2rem auto;
            padding: 0 1rem;
            flex: 1;
        }
        
        .search-card {
            background-color: white;
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            margin-bottom: 2rem;
            position: relative;
        }
        
        h1 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            font-size: 1.6rem;
            margin-bottom: 1.5rem;
            color: #666;
        }
        
        .search-form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            position: relative;
        }
        
        @media (min-width: 768px) {
            .search-form {
                flex-direction: row;
            }
        }
        
        .search-form input {
            flex: 1;
            padding: 0.8rem 1rem;
            border: 2px solid var(--light-gray);
            border-radius: var(--border-radius);
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        .search-form input:focus {
            border-color: var(--primary-color);
            outline: none;
        }
        
        .search-form button {
            padding: 0.8rem 1.5rem;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        
        .search-form button:hover {
            background-color: var(--primary-dark);
        }
        
        .error {
            color: #e74c3c;
            margin-top: 1rem;
            font-weight: 500;
        }
        
        .match-info {
            color: #666;
            font-style: italic;
            margin-bottom: 1rem;
        }
        
        .recommendations {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 2rem;
            box-shadow: var(--box-shadow);
        }
        
        .recommendations h3 {
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--light-gray);
            color: var(--primary-color);
        }
        
        .book-list {
            list-style-type: none;
        }
        
        .book-item {
            padding: 1rem;
            margin-bottom: 0.8rem;
            background-color: #f9f9f9;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--primary-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .book-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        footer {
            text-align: center;
            padding: 1.5rem;
            background-color: var(--primary-color);
            color: white;
            margin-top: auto;
        }
        
        .empty-state {
            text-align: center;
            padding: 3rem 0;
            color: #666;
        }
        
        .empty-state p {
            margin-top: 1rem;
            font-size: 1.1rem;
        }
        
        /* Autocomplete styles */
        .autocomplete-container {
            position: relative;
            flex: 1;
        }
        
        .autocomplete-results {
            position: absolute;
            z-index: 999;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid var(--light-gray);
            border-radius: 0 0 var(--border-radius) var(--border-radius);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            max-height: 300px;
            overflow-y: auto;
            display: none;
        }
        
        .autocomplete-results.active {
            display: block;
        }
        
        .autocomplete-item {
            padding: 0.8rem 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
            border-bottom: 1px solid var(--light-gray);
        }
        
        .autocomplete-item:last-child {
            border-bottom: none;
        }
        
        .autocomplete-item:hover {
            background-color: #f5f5f5;
        }
        
        .highlight {
            font-weight: bold;
            color: var(--primary-color);
        }
    </style>
</head>
<body>
    <header>
        <h1>Book Recommendation System</h1>
        <p>Find your next favorite read based on machine learning</p>
    </header>
    
    <div class="container">
        <div class="search-card">
            <h2>Discover New Books</h2>
            <form class="search-form" method="POST" id="search-form">
                <div class="autocomplete-container">
                    <input 
                        type="text" 
                        name="book_name" 
                        id="book-search" 
                        placeholder="Enter a book title you enjoy..." 
                        value="{{ book_name }}" 
                        autocomplete="off"
                        required
                    >
                    <div class="autocomplete-results" id="autocomplete-results"></div>
                </div>
                <button type="submit">Get Recommendations</button>
            </form>
            
            {% if error %}
                <p class="error">{{ error }}</p>
            {% endif %}
        </div>
        
        {% if recommendations %}
            <div class="recommendations">
                {% if matched_title and matched_title != book_name %}
                    <p class="match-info">Based on: "{{ matched_title }}"</p>
                {% endif %}
                
                <h3>Recommended Books For You</h3>
                <ul class="book-list">
                    {% for book in recommendations %}
                        <li class="book-item">{{ book }}</li>
                    {% endfor %}
                </ul>
            </div>
        {% elif not error and request.method != 'POST' %}
            <div class="empty-state">
                <svg width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 19.5C4 18.837 4.26339 18.2011 4.73223 17.7322C5.20107 17.2634 5.83696 17 6.5 17H20" stroke="#6d28d9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M6.5 2H20V22H6.5C5.83696 22 5.20107 21.7366 4.73223 21.2678C4.26339 20.7989 4 20.163 4 19.5V4.5C4 3.83696 4.26339 3.20107 4.73223 2.73223C5.20107 2.26339 5.83696 2 6.5 2Z" stroke="#6d28d9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p>Enter a book title to get personalized recommendations</p>
            </div>
        {% endif %}
    </div>
    
    <footer>
        <p>© 2025 Book Recommendation System | Powered by Machine Learning</p>
    </footer>
    
    <script>
        // Autocomplete functionality
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('book-search');
            const resultsContainer = document.getElementById('autocomplete-results');
            const searchForm = document.getElementById('search-form');
            
            let selectedIndex = -1;
            let searchResults = [];
            
            // Fetch book suggestions as user types
            searchInput.addEventListener('input', debounce(async function() {
                const query = searchInput.value.trim();
                
                if (query.length < 2) {
                    resultsContainer.classList.remove('active');
                    return;
                }
                
                try {
                    const response = await fetch(`/autocomplete?q=${encodeURIComponent(query)}`);
                    searchResults = await response.json();
                    
                    if (searchResults.length > 0) {
                        displayResults(searchResults, query);
                        resultsContainer.classList.add('active');
                    } else {
                        resultsContainer.classList.remove('active');
                    }
                } catch (error) {
                    console.error('Error fetching autocomplete results:', error);
                }
            }, 300));
            
            // Handle keyboard navigation
            searchInput.addEventListener('keydown', function(e) {
                if (!resultsContainer.classList.contains('active')) return;
                
                const items = resultsContainer.querySelectorAll('.autocomplete-item');
                
                // Down arrow
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                    highlightItem(items);
                }
                
                // Up arrow
                else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    selectedIndex = Math.max(selectedIndex - 1, -1);
                    highlightItem(items);
                }
                
                // Enter key
                else if (e.key === 'Enter' && selectedIndex >= 0) {
                    e.preventDefault();
                    searchInput.value = searchResults[selectedIndex];
                    resultsContainer.classList.remove('active');
                    selectedIndex = -1;
                    searchForm.submit();
                }
                
                // Escape key
                else if (e.key === 'Escape') {
                    resultsContainer.classList.remove('active');
                    selectedIndex = -1;
                }
            });
            
            // Hide results when clicking outside
            document.addEventListener('click', function(e) {
                if (!searchInput.contains(e.target) && !resultsContainer.contains(e.target)) {
                    resultsContainer.classList.remove('active');
                }
            });
            
            // Display results with highlighted matching parts
            function displayResults(results, query) {
                resultsContainer.innerHTML = '';
                
                results.forEach((result, index) => {
                    const item = document.createElement('div');
                    item.className = 'autocomplete-item';
                    
                    // Highlight matching part
                    const regex = new RegExp(`(${escapeRegExp(query)})`, 'gi');
                    const highlightedText = result.replace(regex, '<span class="highlight">$1</span>');
                    
                    item.innerHTML = highlightedText;
                    
                    // Click handler
                    item.addEventListener('click', function() {
                        searchInput.value = result;
                        resultsContainer.classList.remove('active');
                        searchForm.submit();
                    });
                    
                    // Mouseover handler
                    item.addEventListener('mouseover', function() {
                        selectedIndex = index;
                        highlightItem(resultsContainer.querySelectorAll('.autocomplete-item'));
                    });
                    
                    resultsContainer.appendChild(item);
                });
            }
            
            // Highlight selected item
            function highlightItem(items) {
                items.forEach((item, index) => {
                    if (index === selectedIndex) {
                        item.style.backgroundColor = '#f0f0f0';
                    } else {
                        item.style.backgroundColor = '';
                    }
                });
                
                if (selectedIndex >= 0) {
                    const selectedItem = items[selectedIndex];
                    selectedItem.scrollIntoView({ block: 'nearest' });
                }
            }
            
            // Debounce function to limit API calls
            function debounce(func, delay) {
                let timeout;
                return function() {
                    const context = this;
                    const args = arguments;
                    clearTimeout(timeout);
                    timeout = setTimeout(() => func.apply(context, args), delay);
                };
            }
            
            // Escape special characters for regex
            function escapeRegExp(string) {
                return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            }
        });
    </script>
</body>
</html>
'''

# Load data and prepare model
def load_model():
    # Check if file exists
    csv_path = 'books1.csv'
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found!")
        # Use a more common filename as fallback
        alternative_paths = ['books.csv', 'books1.csv', 'books_data.csv']
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                csv_path = alt_path
                print(f"Using alternative CSV file: {alt_path}")
                break
    
    try:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df2 = df.copy()
        
        # Ensure required columns exist
        required_columns = ['title', 'average_rating', 'ratings_count', 'language_code']
        missing_columns = [col for col in required_columns if col not in df2.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert ratings to float if needed
        df2['average_rating'] = df2['average_rating'].astype(float)
        
        # Create rating categories
        df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
        df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
        df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
        df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
        df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"
        
        # Create feature matrix
        rating_df = pd.get_dummies(df2['rating_between'])
        language_df = pd.get_dummies(df2['language_code'])
        features = pd.concat([rating_df, 
                            language_df, 
                            df2['average_rating'], 
                            df2['ratings_count']], axis=1)
        
        # Scale features
        min_max_scaler = MinMaxScaler()
        features = min_max_scaler.fit_transform(features)
        
        # Train model
        model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
        model.fit(features)
        dist, idlist = model.kneighbors(features)
        
        print(f"Model loaded successfully! Dataset has {len(df2)} books")
        return df2, idlist
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Return empty dataframes as fallback
        return pd.DataFrame(), []

# Find matching book titles
def find_matching_books(query, max_results=10):
    if df2 is None or df2.empty:
        return []
        
    query = query.lower().strip()
    
    # Direct partial matching
    direct_matches = df2[df2['title'].str.lower().str.contains(query, case=False, na=False)]
    
    # If we have enough direct matches, return those
    if len(direct_matches) >= max_results:
        return direct_matches['title'].head(max_results).tolist()
    
    # Otherwise, try fuzzy matching for the remaining slots
    result_set = set(direct_matches['title'].tolist())
    remaining_slots = max_results - len(result_set)
    
    if remaining_slots > 0:
        # Get all titles not already in the result set
        other_titles = [t for t in df2['title'].tolist() if t not in result_set]
        
        # Use difflib to find close matches
        fuzzy_matches = difflib.get_close_matches(query, other_titles, n=remaining_slots, cutoff=0.4)
        result_set.update(fuzzy_matches)
    
    return list(result_set)

# Book recommendation function
def book_recommender(book_name):
    global df2, idlist
    
    # Check if model is loaded
    if df2 is None or idlist is None or df2.empty:
        return [], "Error: Book database not loaded properly"
    
    # Step 1: Try substring match (case-insensitive)
    matches = df2[df2['title'].str.contains(book_name, case=False, na=False)]

    if not matches.empty:
        matched_title = matches.iloc[0]['title']
    else:
        # Step 2: Fallback to fuzzy match if no substring match found
        all_titles = df2['title'].tolist()
        closest_match = difflib.get_close_matches(book_name, all_titles, n=1, cutoff=0.5)
        if not closest_match:
            return [], "No similar book title found in our database"
        matched_title = closest_match[0]

    book_list_name = []
    try:
        book_index = df2[df2['title'] == matched_title].index[0]
        
        # Recommend books (skip first one as it's the same book)
        for newid in idlist[book_index][1:6]:  # Get top 5 recommendations
            book_list_name.append(df2.loc[newid].title)
            
        return book_list_name, matched_title
    except IndexError:
        return [], "Error finding the book in our database"

# Autocomplete API endpoint
@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    
    matching_titles = find_matching_books(query)
    return jsonify(matching_titles)

@app.route('/', methods=['GET', 'POST'])
def index():
    matched_title = None
    recommendations = None
    error = None
    
    if request.method == 'POST':
        book_name = request.form.get('book_name', '')
        if book_name:
            recommendations, result = book_recommender(book_name)
            if not recommendations:
                error = result  # This will be the error message
            else:
                matched_title = result  # This will be the matched title
    
    return render_template_string(HTML_TEMPLATE, 
                                recommendations=recommendations, 
                                book_name=request.form.get('book_name', '') if request.method == 'POST' else '',
                                matched_title=matched_title,
                                error=error)

# Try to pre-load the model when the app starts
try:
    print("Loading book recommendation model...")
    df2, idlist = load_model()
except Exception as e:
    print(f"Error pre-loading model: {str(e)}")
    df2, idlist = None, None

if __name__ == '__main__':
    # Check if we loaded the model
    if df2 is None or df2.empty:
        print("WARNING: Book database not loaded! App will show errors when making recommendations.")
    
    # Run the Flask app
    print("Starting Flask app... Access it at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0')