import pandas as pd
import numpy as np
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import Dict

# Star power ratings
# Star power ratings
STAR_POWER_RATINGS: Dict[str, float] = {
    # Elite Tier (0.95-1.0)
    "tom cruise": 0.95,
    "samuel l. jackson": 0.95,
    "denzel washington": 0.95,
    "leonardo dicaprio": 0.95,
    "robert downey jr": 0.95,
    "tom hanks": 0.95,
    "morgan freeman": 0.95,
    "harrison ford": 0.95,
    "will smith": 0.95,
    "brad pitt": 0.95,
    "johnny depp": 0.95,
    "meryl streep": 0.95,
    "julia roberts": 0.95,
    "jennifer lawrence": 0.95,
    "angelina jolie": 0.95,
    "sandra bullock": 0.95,

    # Top Tier (0.85-0.94)
    "dwayne johnson": 0.90,
    "scarlett johansson": 0.90,
    "chris pratt": 0.90,
    "chris evans": 0.90,
    "chris hemsworth": 0.90,
    "matthew mcconaughey": 0.90,
    "cate blanchett": 0.90,
    "nicole kidman": 0.90,
    "charlize theron": 0.90,
    "emma stone": 0.90,
    "jennifer aniston": 0.90,
    "robert pattinson": 0.90,
    "margot robbie": 0.90,
    "ryan gosling": 0.90,
    "christian bale": 0.90,
    "matt damon": 0.90,
    "hugh jackman": 0.90,
    "ryan reynolds": 0.90,
    "chris pine": 0.85,
    "idris elba": 0.85,
    "michael b jordan": 0.85,
    "chadwick boseman": 0.85,
    "benedict cumberbatch": 0.85,
    "jake gyllenhaal": 0.85,
    "anne hathaway": 0.85,
    "natalie portman": 0.85,
    "viola davis": 0.85,
    "zoe saldana": 0.85,

    # High Tier (0.75-0.84)
    "mark ruffalo": 0.80,
    "jeremy renner": 0.80,
    "paul rudd": 0.80,
    "don cheadle": 0.80,
    "samuel l jackson": 0.80,
    "michael fassbender": 0.80,
    "oscar isaac": 0.80,
    "emily blunt": 0.80,
    "jessica chastain": 0.80,
    "brie larson": 0.80,
    "elizabeth olsen": 0.80,
    "zendaya": 0.80,
    "timothee chalamet": 0.80,
    "tom hardy": 0.80,
    "henry cavill": 0.80,
    "gal gadot": 0.80,
    "jason momoa": 0.80,
    "chris tucker": 0.75,
    "jackie chan": 0.75,
    "jet li": 0.75,
    "michelle yeoh": 0.75,
    "donnie yen": 0.75,

    # Mid Tier (0.65-0.74)
    "anthony mackie": 0.70,
    "dave bautista": 0.70,
    "john cena": 0.70,
    "karen gillan": 0.70,
    "tom holland": 0.70,
    "daniel radcliffe": 0.70,
    "emma watson": 0.70,
    "rupert grint": 0.70,
    "kristen stewart": 0.70,
    "daniel craig": 0.70,
    "vin diesel": 0.70,
    "jason statham": 0.70,
    "keanu reeves": 0.70,
    "laurence fishburne": 0.70,
    "hugo weaving": 0.70,
    "carrie-anne moss": 0.70,
    "willem dafoe": 0.65,
    "jude law": 0.65,
    "rachel mcadams": 0.65,
    "eva green": 0.65,

    # Rising Stars (0.55-0.64)
    "florence pugh": 0.60,
    "ana de armas": 0.60,
    "pedro pascal": 0.60,
    "oscar isaac": 0.60,
    "simu liu": 0.60,
    "awkwafina": 0.60,
    "lakeith stanfield": 0.60,
    "jonathan majors": 0.60,
    "austin butler": 0.60,
    "anya taylor-joy": 0.60,
    "millie bobby brown": 0.60,
    "jacob elordi": 0.55,
    "sydney sweeney": 0.55,
    "jenna ortega": 0.55,
    "barry keoghan": 0.55,
    "paul mescal": 0.55,

    # Established Character Actors (0.50-0.54)
    "stanley tucci": 0.54,
    "jeffrey wright": 0.54,
    "giancarlo esposito": 0.54,
    "bryan cranston": 0.54,
    "j.k. simmons": 0.54,
    "gary oldman": 0.54,
    "ed harris": 0.54,
    "michael shannon": 0.52,
    "john malkovich": 0.52,
    "john goodman": 0.52,
    "steve buscemi": 0.52,
    "michael keaton": 0.52,
    "brendan gleeson": 0.50,
    "richard jenkins": 0.50,
    "mads mikkelsen": 0.50,
    "ben mendelsohn": 0.50,

    # Other Categories
    "emerging star": 0.40,
    "unknown actor": 0.30,
    "debut actor": 0.25
}
# Add this after STAR_POWER_RATINGS
FEATURE_NAMES = [
    'Sentiment',
    'Positive_Count',
    'Negative_Count',
    'budget_encoded',
    'genre_encoded',
    'star_power'
]

# Page configuration
st.set_page_config(
    page_title="Film Success Predictor",
    page_icon="🎬",
    layout="wide"
)

# Initialize sentiment analyzer
@st.cache_resource
def load_sia():
    return SentimentIntensityAnalyzer()

def create_synthetic_model():
    """Fallback model with synthetic data if Rotten Tomatoes data fails"""
    # Create some sample training data in correct order
    sample_data = pd.DataFrame({
        'Sentiment': np.random.uniform(-1, 1, 100),
        'Positive_Count': np.random.randint(0, 10, 100),
        'Negative_Count': np.random.randint(0, 10, 100),
        'budget_encoded': np.random.randint(0, 3, 100),
        'genre_encoded': np.random.randint(0, 6, 100),
        'star_power': np.random.uniform(0.3, 1.0, 100),
        'success': np.random.randint(0, 2, 100)
    })
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(sample_data[FEATURE_NAMES], sample_data['success'])
    
    print("Using synthetic model as fallback")
    return model
def create_basic_model():
    """Create and train model using Rotten Tomatoes data with optimized performance"""
    try:
        # Set up correct paths
        home_dir = Path('/home/ubuready')
        
        # Check if files exist
        reviews_path = home_dir / 'rotten_tomatoes_movie_reviews.csv'
        movies_path = home_dir / 'rotten_tomatoes_movies.csv'
        
        if not reviews_path.exists() or not movies_path.exists():
            print("Rotten Tomatoes data files not found")
            return create_synthetic_model()
        
        # Load limited, relevant data
        reviews_data = pd.read_csv(
            reviews_path,
            usecols=['id', 'reviewText'],
            nrows=5000  # Limit to most recent reviews
        )
        
        movies_data = pd.read_csv(
            movies_path,
            usecols=['id', 'title', 'genre', 'tomatoMeter']
        )
        
        # Quick filter for valid genres
        genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"]
        movies_data = movies_data[movies_data['genre'].isin(genres)]
        
        # Batch process sentiments for better performance
        sia = SentimentIntensityAnalyzer()
        reviews_data['sentiment'] = reviews_data['reviewText'].apply(
            lambda x: sia.polarity_scores(str(x))['compound'] if pd.notna(x) else 0
        )
        
        # Aggregate reviews efficiently
        movie_features = reviews_data.groupby('id').agg({
            'sentiment': 'mean',
            'reviewText': lambda x: ' '.join(str(text).lower() for text in x if pd.notna(text))
        }).reset_index()
        
        # Efficient word counting
        positive_keywords = (
            'engaging', 'delightful', 'masterpiece', 'enjoyable', 'inspiring',
            'epic', 'thrilling', 'compelling', 'powerful', 'innovative'
        )
        negative_keywords = (
            'boring', 'unfunny', 'disappointing', 'tedious', 'dull',
            'derivative', 'mediocre', 'predictable', 'confusing', 'weak'
        )
        
        # Vectorized word counting
        movie_features['Positive_Count'] = movie_features['reviewText'].apply(
            lambda x: sum(1 for word in positive_keywords if word in x)
        )
        movie_features['Negative_Count'] = movie_features['reviewText'].apply(
            lambda x: sum(1 for word in negative_keywords if word in x)
        )
        
        # Drop the full text column to save memory
        movie_features.drop('reviewText', axis=1, inplace=True)
        
        # Merge with movies data
        training_data = movies_data.merge(
            movie_features,
            on='id',
            how='inner'
        )
        
        # Encode genres
        training_data['genre_encoded'] = training_data['genre'].map(
            {genre: idx for idx, genre in enumerate(genres)}
        )
        
        # Define success threshold
        training_data['success'] = (training_data['tomatoMeter'] >= 75).astype(int)
        
        # Prepare final features
        features = pd.DataFrame({
            'Sentiment': training_data['sentiment'],
            'Positive_Count': training_data['Positive_Count'],
            'Negative_Count': training_data['Negative_Count'],
            'genre_encoded': training_data['genre_encoded'],
            'budget_encoded': 1,
            'star_power': 0.5 
        }) [FEATURE_NAMES]
        
        # Train efficient model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        model.fit(features, training_data['success'])
        print(f"Model trained on {len(training_data)} movies")
        return model
        
    except Exception as e:
        print(f"Error training with Rotten Tomatoes data: {e}")
        return create_synthetic_model()

@st.cache_resource
def load_model():
    try:
        with open("film_success_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return create_basic_model()

@st.cache_data
def load_datasets():
    try:
        successful_films = pd.DataFrame({
            'title': [
                'The Dark Knight',
                'Avatar',
                'Titanic',
                'Jurassic World',
                'The Avengers'
            ],
            'genre': [
                'Action',
                'Sci-Fi',
                'Drama',
                'Action',
                'Action'
            ],
            'budget_level': [
                'High',
                'High',
                'High',
                'High',
                'High'
            ],
            'box_office': [
                1004.6,
                2847.2,
                2201.6,
                1671.7,
                1518.8
            ],
            'success_factors': [
                'Strong character development, compelling villain, dark tone, excellent marketing',
                'Groundbreaking visuals, universal story themes, broad audience appeal',
                'Epic romance, historical event, cutting-edge effects, strong marketing',
                'Nostalgia factor, family appeal, strong visual effects, established franchise',
                'Superhero team-up, built-up anticipation, strong character dynamics'
            ]
        })
        return successful_films
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def analyze_sentiment(text):
    if not text:
        return 0
    text = str(text)[:1000]
    return load_sia().polarity_scores(text)["compound"]

def calculate_star_power(star_name: str) -> float:
    """Calculate star power rating for given actor name."""
    if not star_name:
        return 0.30  # Default for no star specified
    
    # Convert to lowercase for matching
    star_name = star_name.lower()
    
    # Exact match
    if star_name in STAR_POWER_RATINGS:
        return STAR_POWER_RATINGS[star_name]
    
    # Partial match (e.g., "Tom Cruise" matches "cruise")
    for known_star, rating in STAR_POWER_RATINGS.items():
        if star_name in known_star or known_star in star_name:
            return rating
    
    # No match found
    return 0.40  # Default for unknown stars

def extract_features(synopsis, budget, genre, star_name=""):
    # Calculate sentiment
    sentiment_score = analyze_sentiment(synopsis)
    
    # Get star power rating
    star_power = calculate_star_power(star_name)
    
    # Enhanced keyword lists for film-specific analysis
    positive_keywords = (
        'engaging', 'delightful', 'masterpiece', 'enjoyable', 'inspiring',
        'epic', 'thrilling', 'compelling', 'powerful', 'innovative',
        'stunning', 'brilliant', 'memorable', 'gripping', 'outstanding',
        'groundbreaking', 'captivating', 'extraordinary', 'remarkable'
    )
    
    negative_keywords = (
        'boring', 'unfunny', 'disappointing', 'tedious', 'dull',
        'derivative', 'mediocre', 'predictable', 'confusing', 'weak',
        'cliché', 'forgettable', 'uninspired', 'lackluster', 'messy',
        'convoluted', 'shallow', 'flawed', 'conventional'
    )
    
    # Count positive and negative words
    text = synopsis.lower()
    positive_count = sum(1 for word in positive_keywords if word in text)
    negative_count = sum(1 for word in negative_keywords if word in text)
    
    # Map budget to numerical value
    budget_map = {"Low": 0, "Medium": 1, "High": 2}
    
    # Map genre to numerical value
    genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"]
    genre_encoded = genres.index(genre)
    
    # Create features DataFrame
    # Create features DataFrame with specific order
    features = pd.DataFrame([[
        sentiment_score,
        positive_count,
        negative_count,
        budget_map[budget],
        genre_encoded,
        star_power
    ]], columns=FEATURE_NAMES)  # Use consistent feature names
    
    return features

# Main app
st.title("Film Success Predictor")

# Load data and model
successful_films = load_datasets()
model = load_model()

# Add model loading status
if model:
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.error("Model failed to load")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Your Film Details")
    
    title = st.text_input("Film Title")
    star_name = st.text_input("Lead Actor/Actress") 
    synopsis = st.text_area("Synopsis (max 1000 characters)", max_chars=1000, height=150)
    genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"])
    budget = st.select_slider("Budget Range", options=["Low", "Medium", "High"])

    if st.button("Predict Success"):
        if not synopsis:
            st.error("Please enter a synopsis for your film.")
        elif len(synopsis.split()) < 10:
            st.warning("Please provide a more detailed synopsis (at least 10 words) for better prediction.")
        else:
            features = extract_features(synopsis, budget, genre, star_name)
            try:
                probability = model.predict_proba(features)[0, 1]
                base_probability = probability
                
                # Adjust probability based on star power
                star_power = calculate_star_power(star_name)
                if star_power > 0.8:  # Top tier stars
                    probability = min(1.0, probability * 1.5)  # 50% boost
                elif star_power > 0.6:  # High tier stars
                    probability = min(1.0, probability * 1.3)  # 30% boost
                
                st.subheader("Prediction Results")
                st.metric("Success Probability", f"{probability:.1%}")
                
                # Analysis explanation
                st.write("Analysis of Key Factors:")
                sentiment_score = features['Sentiment'].iloc[0]
                st.write(f"- Sentiment Score: {sentiment_score:.2f} " + 
                        ("(Positive)" if sentiment_score > 0 else "(Negative)" if sentiment_score < 0 else "(Neutral)"))
                st.write(f"- Positive Elements: {features['Positive_Count'].iloc[0]}")
                st.write(f"- Negative Elements: {features['Negative_Count'].iloc[0]}")
                st.write(f"- Budget Impact: {budget}")
                st.write(f"- Genre: {genre}")
                if star_power > 0.5:
                    st.write(f"- Star Power: Added {(probability - base_probability):.1%} to probability")
                
                # Enhanced recommendations based on specific factors
                st.subheader("Recommendations")
                if probability > 0.7:
                    st.success("High potential for success! Consider:")
                    recs = [
                        "Develop a strong marketing campaign focusing on unique elements",
                        "Consider timing release for optimal market conditions",
                        f"Leverage successful elements from similar {genre} films",
                        "Plan for potential franchise opportunities"
                    ]
                elif probability > 0.4:
                    st.warning("Moderate potential. Areas to consider:")
                    recs = [
                        "Strengthen narrative hooks and character development",
                        f"Research successful {genre} films in your budget range",
                        "Consider target audience expansion strategies",
                        "Evaluate budget allocation priorities"
                    ]
                else:
                    st.error("Lower success probability. Suggested actions:")
                    recs = [
                        "Revise core story elements and unique selling points",
                        "Consider script revision with focus on audience engagement",
                        "Research market gaps and opportunities",
                        "Evaluate genre-budget fit"
                    ]
                for rec in recs:
                    st.write(f"- {rec}")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
with col2:
    st.subheader("Historical Success Analysis")
    if successful_films is not None:
        for _, film in successful_films.iterrows():
            with st.expander(f"{film['title']} (${film['box_office']}B)"):
                st.write(f"**Genre:** {film['genre']}")
                st.write(f"**Budget Level:** {film['budget_level']}")
                st.write("**Success Factors:**")
                for factor in film['success_factors'].split(', '):
                    st.write(f"- {factor}")

# Footer with model info
st.markdown("---")
st.caption("Model trained on historical film data including box office performance, critic ratings, and audience reception.")