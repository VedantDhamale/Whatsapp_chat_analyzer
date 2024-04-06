import re
import pandas as pd

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m\s-\s'

    # Split the data into messages and dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    messages = messages[:len(dates)]
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    users = []
    messages_list = []

    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages_list.append(entry[2])
        else:
            users.append('group_notification')
            messages_list.append(entry[0])

    # Create new columns in the DataFrame
    df['user'] = users
    df['message'] = messages_list

    # Drop the original 'user_message' column
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year.astype(str)
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.strftime('%I %p')  # Format hour in AM/PM format
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == '11 PM' or hour == '12 AM':
            period.append(hour + "-" + str('01 AM'))
        elif hour == '12 PM':
            period.append(hour + "-" + str('01 PM'))
        else:
            hour_int = int(hour.split()[0])
            period.append(hour + "-" + str(hour_int + 1) + " " + hour.split()[1])
    df['period'] = period

    # Analyze sentiment using TextBlob
    #df['sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df
