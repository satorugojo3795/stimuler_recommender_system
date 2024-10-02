import math
import random
import numpy as np
import datetime
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict #provides default value rather than raising keyError

@dataclass
class Error:
    category: str #main category of the error ( grammar, vocabulary etc)
    subcategory: str #(tenses, preposition etc)
    timestamp: datetime.datetime
    
@dataclass
class ExerciseStats:
    total_attempts: int = 0
    successes: int = 0
    ratings: List[int] = field(default_factory=list)  # a new list, is created fresh for each instance of a class, rather than shared among instances.

#class to store the exercises and the various things that an exercise can have
@dataclass
class Exercise:
    exercise_id: str
    category: str
    skill_tags: List[str]
    difficulty_level: str
    content: str
    media_type: str
    theme: List[str]
    demographics: Dict[str, List[str]]
    statistics: ExerciseStats = field(default_factory=ExerciseStats) # this thing is being created from the ExerciseStats class
    
#implement user calss
@dataclass
class User:
    user_id: str
    country: str
    age_band: str
    proficiency_level: str
    interests: List[str]
    preferred_learning_style: str
    motivation: str
    error_history: List[Error] = field(default_factory=list) # List of Error instances representing the user's past mistakes. This is made from the error class
    exercise_history: List[Exercise] = field(default_factory=list) # List of Exercise instances the user has completed. This is made from the exercise class
    
#helper function
# Helper Functions
def current_time():
    """Returns the current time

    Returns:
        _type_: datetime.datetime
    """
    return datetime.datetime.now()

# since we want to give more weights to recent error we implemetn exponential decay. It will help us calculate a weight based on how much time has passed since an error occurred
def exponential_decay(time_diff_seconds, decay_rate=0.0001):
    return math.exp(-decay_rate * time_diff_seconds)

# we may want to recommend similar exercise to users who have similar data(country, demographics etc) for this we implement jaccard similarity
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# global variables
exercise_stats = defaultdict(ExerciseStats)

# Per-user Exercise Statistics
exercise_stats_per_user = defaultdict(ExerciseStats)

# error analysts and priority calculation scores

def calculate_priority_scores(user: User):
    """this function process the user's history to calculate priority scores for each category and skill. By weighing errors based on recency and frequency, we can identify the areas where user need to focus on and thus will guide the recommendation system to focus on the most relevant content for the user
    """
    current = current_time()
    category_scores = {}
    skill_scores = {}
    
    for error in user.error_history:
        time_diff = (current - error.timestamp).total_seconds()
        weight = exponential_decay(time_diff)
        
        #update category scores
        category_scores[error.category] = category_scores.get(error.category, 0) + weight
        
        skill_key = (error.category, error.subcategory)
        #update skill scores
        skill_scores[skill_key] = skill_scores.get(skill_key, 0) + weight
        
    return category_scores, skill_scores


# now we need to select the category which has the highest priority score so that it can be recommended by our system
def select_category(category_scores):
    """
    Given a dictionary of category scores, select the category with the highest priority score.
    If no errors are present, the default category is 'grammar'.
    :param category_scores: A dictionary mapping category names to their respective scores.
    :return: The category with the highest priority score.
    """
    if not category_scores:
        return 'grammar'  # Default category if no errors are present
    return max(category_scores, key=category_scores.get)

# now we know the category but we need to address the skills with twhich the user struggles the most and recommend exrecise based on that

def select_skills(skill_scores, top_category):
    
    """
    Given a dictionary of skill scores and the top category, return a list of
    the top 3 skills that the user struggles with the most. The skills are
    filtered to only include those belonging to the top category, sorted by
    score in descending order, and the top 3 skills are extracted.
    """
    
    # Filter skills that belong to the top category
    # {(error.category,error.subcategory):score} - format of skill scores
    # here we will filter out skill scores to only keep the category which is the top category
    relevant_skills = {k: v for k, v in skill_scores.items() if k[0] == top_category}
    if not relevant_skills:
        return []
    # Sort skills by score in descending order of score
    sorted_skills = sorted(relevant_skills.items(), key=lambda item: item[1], reverse=True)
    # Extract the top 3 skills
    top_skills = [skill for (category, skill), score in sorted_skills[:3]]
    return top_skills

# Content Personalization
# To map the user's preferred learning style to the appropriate media types. This helps in filtering exercises that match the user's preferred modality of learning.
def media_type_mapping(learning_style):
    """
    Maps the user's preferred learning style to the appropriate media types.
    
    Parameters:
    learning_style (str): The user's preferred learning style.
    
    Returns:
    List[str]: A list of media types that match the user's preferred learning style.
    """
    return {
        'visual': ['image', 'video'],
        'auditory': ['audio', 'video'],
        'kinesthetic': ['interactive', 'game'],
        'reading/writing': ['text']
    }.get(learning_style, ['text'])
    
#now we check if the demographics of te exercise matches witht the demographics of the user
def demographics_match(ex_demographics, user):
    """
    Checks if the demographics specified in the exercise match the user's demographics.

    Parameters:
    ex_demographics (dict): A dictionary containing the demographics specified in the exercise.
    user (User): The user object.

    Returns:
    bool: A boolean indicating whether the demographics match.
    """
    country_match = 'All' in ex_demographics.get('countries', []) or user.country in ex_demographics.get('countries', [])
    age_match = 'All' in ex_demographics.get('age_bands', []) or user.age_band in ex_demographics.get('age_bands', [])
    return country_match and age_match

#now we can filter down the exercises to a list of exercises which best fit the user's needs, preferences, and demographics

def filter_exercises(exercises: List[Exercise], category: str, skills: List[str], user: User):
    """
    Filter exercises based on user preferences and demographics.
    
    Parameters:
    exercises (List[Exercise]): The list of exercises to filter.
    category (str): The category to filter by.
    skills (List[str]): The list of skills to filter by.
    user (User): The user object.
    
    Returns:
    List[Exercise]: A list of exercises which best fit the user's needs, preferences, and demographics.
    """
    filtered = []
    for ex in exercises:
        if ex.category != category:
            continue
        if ex.difficulty_level != user.proficiency_level:
            continue
        if skills and not set(ex.skill_tags).intersection(skills):
            continue
        if ex.media_type not in media_type_mapping(user.preferred_learning_style):
            continue
        if not set(ex.theme).intersection(user.interests):
            continue
        if not demographics_match(ex.demographics, user):
            continue
        filtered.append(ex)
    return filtered

# but what if no exercise matches a user's preference then what should we do? In this case we will return a default exercise

# for such cases we only check for the category, diffiultuy level and user's proficiency level

def default_exercises(exercises, category, user):
    return [ex for ex in exercises if ex.category == category and ex.difficulty_level == user.proficiency_level]

#implementing advanced functions

# for users wo have similar interests or errors we can recommend similar exercises based on collaborative filtering
# Similarity and Collaborative Filtering

def compute_user_similarity(user1: User, user2: User):
    """
    Computes similarity between two users based on interests and error patterns.

    Args:
        user1 (User): The first user.
        user2 (User): The second user.

    Returns:
        float: A similarity score between 0 and 1.
    """
    interests_similarity = jaccard_similarity(set(user1.interests), set(user2.interests))
    
    error_similarity = compute_error_similarity(user1, user2)
    
    # Combine the similarities with equal weighting
    return 0.5 * interests_similarity + 0.5 * error_similarity

#now we implement the compute_error_similarity
def compute_error_similarity(user1: User, user2: User):
    """
    Computes similarity between two users based on their error histories.

    Args:
        user1 (User): The first user.
        user2 (User): The second user.

    Returns:
        float: A similarity score between 0 and 1.
    """
    user1_errors = {(error.category, error.subcategory) for error in user1.error_history}
    user2_errors = {(error.category, error.subcategory) for error in user2.error_history}
    return jaccard_similarity(user1_errors, user2_errors)


def find_similar_users(user: User, all_users: List[User], top_n: int = 10):
    """
    Finds the top N users most similar to the given user.

    Args:
        user (User): The user for whom to find similar users.
        all_users (List[User]): A list of all users.
        top_n (int): The number of similar users to return.

    Returns:
        List[User]: A list of similar users.
    """
    similarities = []
    for other_user in all_users:
        if other_user.user_id == user.user_id:
            continue  # because same user
        similarity = compute_user_similarity(user, other_user)
        similarities.append((similarity, other_user))
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    # Return the top N similar users
    similar_users = [user for sim, user in similarities[:top_n]]
    return similar_users

# now we can implement a function which can adjust the reward of an exercise based on how effective it has been for similar users

def collaborative_filtering_score(exercise: Exercise, similar_users: List[User]):
    """
    Computes a collaborative filtering score for an exercise based on similar users.

    Args:
        exercise (Exercise): The exercise to evaluate.
        similar_users (List[User]): A list of users similar to the current user.

    Returns:
        float: A score between 0 and 1 representing the expected effectiveness of the exercise.
    """
    total_attempts = 0
    total_successes = 0
    total_ratings = []

    for user in similar_users:
        key = (user.user_id, exercise.exercise_id)
        stats = exercise_stats_per_user.get(key)
        if stats:
            total_attempts += stats.total_attempts
            total_successes += stats.successes
            total_ratings.extend(stats.ratings)

    if total_attempts == 0:
        return 0.5  # Neutral score if no data

    success_rate = total_successes / total_attempts
    avg_rating = np.mean(total_ratings) / 5 if total_ratings else 0.5

    # Weighted combination of success rate and average rating
    cf_score = 0.7 * success_rate + 0.3 * avg_rating
    return cf_score


# To introduce variety in the recommendations by avoiding themes the user has recently seen, we implement ensure_diversity function
def ensure_diversity(exercises: List[Exercise], user: User):
    """
    Filters exercises to ensure content diversity by avoiding recently used themes.

    Args:
        exercises (List[Exercise]): A list of candidate exercises.
        user (User): The user for whom to filter exercises.

    Returns:
        List[Exercise]: A list of exercises that introduce new themes.
    """
    # Collect themes from the user's recent exercise history
    recent_themes = {theme for ex in user.exercise_history[-5:] for theme in ex.theme}
    # Find exercises with themes not in recent themes
    diverse_exercises = [ex for ex in exercises if not set(ex.theme).intersection(recent_themes)]
    # If there are exercises with new themes, return them
    if diverse_exercises:
        return diverse_exercises
    # Otherwise, return the original list
    return exercises

def select_best_exercise(exercises: List[Exercise], user: User, all_users: List[User]):
    """
    Selects the best exercise using Thompson Sampling and Collaborative Filtering.

    Args:
        exercises (List[Exercise]): A list of candidate exercises.
        user (User): The user for whom to select the exercise.
        all_users (List[User]): A list of all users for collaborative filtering.

    Returns:
        Exercise: The selected exercise.
    """
    candidate_exercises = []

    # Find similar users
    similar_users = find_similar_users(user, all_users)

    # Calculate expected rewards using Bayesian inference and collaborative filtering
    for exercise in exercises:
        stats = exercise_stats[exercise.exercise_id]
        
        # Prior parameters for Beta distribution
        alpha = 1 + stats.successes
        beta_param = 1 + stats.total_attempts - stats.successes

        # Sample from Beta distribution to get expected reward (Thompson Sampling)
        sampled_reward = np.random.beta(alpha, beta_param)

        # Collaborative filtering adjustment
        cf_score = collaborative_filtering_score(exercise, similar_users)

        # Final score combining sampled reward and collaborative filtering score
        final_score = 0.6 * sampled_reward + 0.4 * cf_score

        candidate_exercises.append((final_score, exercise))

    # Sort exercises based on the final score in descending order
    candidate_exercises.sort(key=lambda x: x[0], reverse=True)

    # Select top N exercises to introduce diversity
    N = min(5, len(candidate_exercises))
    top_exercises = [exercise for score, exercise in candidate_exercises[:N]]

    # Ensure diversity in the selection
    diverse_exercises = ensure_diversity(top_exercises, user)

    # Select the final exercise
    selected_exercise = diverse_exercises[0] if diverse_exercises else top_exercises[0]

    return selected_exercise

# Recommendation Engine

def recommend_exercise(user: User, exercises: List[Exercise], all_users: List[User]):
    """
    Recommends an exercise to the user based on their error history and preferences.

    Args:
        user (User): The user for whom to recommend an exercise.
        exercises (List[Exercise]): A list of all available exercises.
        all_users (List[User]): A list of all users.

    Returns:
        Exercise: The recommended exercise.
    """
    # Error analysis
    category_scores, skill_scores = calculate_priority_scores(user)
    top_category = select_category(category_scores)
    top_skills = select_skills(skill_scores, top_category)
    # Content personalization
    suitable_exercises = filter_exercises(exercises, top_category, top_skills, user)
    if not suitable_exercises:
        # Use default exercises if no suitable ones are found
        suitable_exercises = default_exercises(exercises, top_category, user)
    # Advanced exercise selection
    selected_exercise = select_best_exercise(suitable_exercises, user, all_users)
    return selected_exercise


# now we update global exercise statistics after a user completes an exercise, reflecting how effective the exercise is overall. This data is essential for the recommendation system to learn from user interactions and adjust future recommendations.

# Updating Exercise Statistics

def update_exercise_stats(exercise_id: str, success: bool, rating: int = None):
    """
    Updates the global exercise statistics after a user completes an exercise.

    Args:
        exercise_id (str): The ID of the exercise.
        success (bool): Indicates whether the user improved after the exercise.
        rating (int, optional): The user's feedback rating for the exercise (1-5). Defaults to None.
    """
    stats = exercise_stats[exercise_id]
    stats.total_attempts += 1
    stats.successes += int(success)
    if rating is not None:
        stats.ratings.append(rating)
        
# after updating the exercise statistics we also need to update the statistics of an individual user. This data is essential for the recommendation system to learn from user interactions and adjust future recommendations.

def update_exercise_stats_per_user(user_id: str, exercise_id: str, success: bool, rating: int = None):
    """
    Updates the per-user exercise statistics after a user completes an exercise.

    Args:
        user_id (str): The ID of the user.
        exercise_id (str): The ID of the exercise.
        success (bool): Indicates whether the user improved after the exercise.
        rating (int, optional): The user's feedback rating for the exercise (1-5). Defaults to None.
    """
    key = (user_id, exercise_id)
    stats = exercise_stats_per_user[key]
    stats.total_attempts += 1
    stats.successes += int(success)
    if rating is not None:
        stats.ratings.append(rating)

# Main Function

def main():
    # Initialize users
    user1 = User(
        user_id='user_123',
        country='Japan',
        age_band='18-24',
        proficiency_level='B1',
        interests=['anime', 'sports'],
        preferred_learning_style='visual',
        motivation='travel'
    )

    user2 = User(
        user_id='user_456',
        country='India',
        age_band='25-34',
        proficiency_level='B2',
        interests=['sitcoms', 'sports'],
        preferred_learning_style='auditory',
        motivation='work'
    )

    # Simulate errors for user1
    user1.error_history.extend([
        Error(category='grammar', subcategory='prepositions', timestamp=current_time()),
        Error(category='vocabulary', subcategory='phrasal_verbs', timestamp=current_time()),
    ])

    # Simulate errors for user2
    user2.error_history.extend([
        Error(category='pronunciation', subcategory='consonant_clusters', timestamp=current_time()),
        Error(category='grammar', subcategory='tenses', timestamp=current_time()),
    ])

    # Create a list of all users
    all_users = [user1, user2]

    # Sample exercises
    exercises = [
        Exercise(
            exercise_id='ex_001',
            category='grammar',
            skill_tags=['prepositions'],
            difficulty_level='B1',
            content='Fill in the blanks with the correct prepositions.',
            media_type='image',
            theme=['anime','sports'],
            demographics={'countries': ['Japan', 'All'], 'age_bands': ['18-24', 'All']}
        ),
        Exercise(
            exercise_id='ex_002',
            category='vocabulary',
            skill_tags=['phrasal_verbs'],
            difficulty_level='B1',
            content='Match the phrasal verbs to their meanings.',
            media_type='video',
            theme=['technology','music'],
            demographics={'countries': ['All'], 'age_bands': ['All']}
        ),
        Exercise(
            exercise_id='ex_003',
            category='pronunciation',
            skill_tags=['consonant_clusters'],
            difficulty_level='B2',
            content='Practice pronouncing these consonant clusters.',
            media_type='audio',
            theme=['music','movie'],
            demographics={'countries': ['India', 'All'], 'age_bands': ['25-34', 'All']}
        ),
        Exercise(
            exercise_id='ex_004',
            category='grammar',
            skill_tags=['tenses'],
            difficulty_level='B2',
            content='Choose the correct tense for each sentence.',
            media_type='text',
            theme=['sitcoms','video'],
            demographics={'countries': ['India', 'All'], 'age_bands': ['25-34', 'All']}
        ),
    ]

    # Recommend exercise for user1
    recommended_exercise_user1 = recommend_exercise(user1, exercises, all_users)
    print(f"\nRecommended Exercise for {user1.user_id}:")
    print(f"Exercise ID: {recommended_exercise_user1.exercise_id}")
    print(f"Category: {recommended_exercise_user1.category}")
    print(f"Content: {recommended_exercise_user1.content}")
    print(f"Theme: {recommended_exercise_user1.theme}")

    # Simulate user1 completing the exercise
    success = True  # Assume user improved
    rating = 5  # User's feedback rating
    update_exercise_stats(recommended_exercise_user1.exercise_id, success, rating)
    update_exercise_stats_per_user(user1.user_id, recommended_exercise_user1.exercise_id, success, rating)
    user1.exercise_history.append(recommended_exercise_user1)

    # Recommend exercise for user2
    recommended_exercise_user2 = recommend_exercise(user2, exercises, all_users)
    print(f"\nRecommended Exercise for {user2.user_id}:")
    print(f"Exercise ID: {recommended_exercise_user2.exercise_id}")
    print(f"Category: {recommended_exercise_user2.category}")
    print(f"Content: {recommended_exercise_user2.content}")
    print(f"Theme: {recommended_exercise_user2.theme}")

    # Simulate user2 completing the exercise
    success = False  # Assume user did not improve
    rating = 3  # User's feedback rating
    update_exercise_stats(recommended_exercise_user2.exercise_id, success, rating)
    update_exercise_stats_per_user(user2.user_id, recommended_exercise_user2.exercise_id, success, rating)
    user2.exercise_history.append(recommended_exercise_user2)

if __name__ == "__main__":
    main()


