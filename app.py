import matplotlib
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emojis

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Sentiment Analysis Function
    def analyze_sentiment_vader(message):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(message)

        # Determine sentiment based on compound score
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply Sentiment Analysis to Messages
    df['sentiment'] = df['message'].apply(analyze_sentiment_vader)

    # Your existing code continues below...
    # Display results including sentiment analysis

    # fetch unique user
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links shared")
            st.title(num_links)

            # monthly TIME LINE
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)

        fig = px.bar(timeline, x='time', y='message', color='message',
                     labels={'time': 'Time', 'message': 'Message Count'})

        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Activity map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group
        if selected_user == 'Overall':
            st.title('Most Busy users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # wordcloud
        st.title("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        plt.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title("Most common words")
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        st.title('Distribution of Sentiment Labels')
        sentiment_counts = df['sentiment'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        st.pyplot(plt)
