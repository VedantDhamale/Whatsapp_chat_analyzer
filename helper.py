from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import re
from textblob import TextBlob  # Import TextBlob for sentiment analysis

# Initialize URL extractor
extract = URLExtract()

# Function to clean text
def clean_text(text):
    # Remove punctuation and special symbols except emojis
    cleaned_text = re.sub(r'[^\w\s😀-🙏]', '', text)
    return cleaned_text


# Function to fetch statistics
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        cleaned_message = clean_text(message)  # Clean the message text
        words.extend(cleaned_message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

# Function to analyze sentiment using TextBlob
def analyze_sentiment_vader(message):
    blob = TextBlob(message)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Modified create_wordcloud function to include sentiment analysis
def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stopwords = f.read().splitlines()  # Read stopwords as a list

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stopwords:
                y.append(word)
        return " ".join(y)

    # Analyze sentiment and clean text for each message
    temp['message_with_sentiment'] = temp['message'].apply(lambda x: f"{clean_text(x)} ({analyze_sentiment_vader(x)})")
    cleaned_messages = ' '.join(temp['message_with_sentiment'])

    # Generate WordCloud based on cleaned text
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(cleaned_messages)
    return df_wc


def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stopwords = f.read().splitlines()  # Read stopwords as a list
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        cleaned_message = clean_text(message)  # Clean the message text
        words.extend([word for word in cleaned_message.lower().split() if word not in stopwords])

    most_common_df = pd.DataFrame(Counter(words).most_common(25))

    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend(c for c in message if c in emoji.EMOJI_DATA)
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline (selected_user , df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []

    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " " + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline

def get_activity_data(selected_user, df):
    # Your logic to retrieve activity data goes here
    # For example, assuming 'activity_category' and 'count' are columns in your DataFrame
    activity_data = df.groupby('activity_category')['count'].sum().reset_index()
    return activity_data
def daily_timeline (selected_user , df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map (selected_user , df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user , df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user , df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


