# 💪 Hand, Arm & Leg Emoji Sentiment Analysis with TextBlob

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

# Step 2: Emoji Dataset
body_emojis = [
    {"Emoji": "🤝", "Name": "Handshake", "Category": "Hand", "Example": "Nice to meet you 🤝"},
    {"Emoji": "✋", "Name": "Raised Hand", "Category": "Hand", "Example": "Stop right there ✋"},
    {"Emoji": "🖐️", "Name": "Hand with Fingers Splayed", "Category": "Hand", "Example": "Hello! 🖐️"},
    {"Emoji": "🖖", "Name": "Vulcan Salute", "Category": "Hand", "Example": "Live long and prosper 🖖"},
    {"Emoji": "👋", "Name": "Waving Hand", "Category": "Hand", "Example": "Hey there! 👋"},
    {"Emoji": "👏", "Name": "Clapping Hands", "Category": "Hand", "Example": "Well done! 👏"},
    {"Emoji": "🙌", "Name": "Raising Hands", "Category": "Hand", "Example": "Yay! 🙌"},
    {"Emoji": "👐", "Name": "Open Hands", "Category": "Hand", "Example": "Come here 👐"},
    {"Emoji": "🙏", "Name": "Folded Hands", "Category": "Hand", "Example": "Thank you 🙏"},
    {"Emoji": "👍", "Name": "Thumbs Up", "Category": "Hand", "Example": "Great job 👍"},
    {"Emoji": "👎", "Name": "Thumbs Down", "Category": "Hand", "Example": "That was bad 👎"},
    {"Emoji": "👊", "Name": "Fist Bump", "Category": "Hand", "Example": "Let's go 👊"},
    {"Emoji": "✊", "Name": "Raised Fist", "Category": "Hand", "Example": "Power to the people ✊"},
    {"Emoji": "🤛", "Name": "Left-Facing Fist", "Category": "Hand", "Example": "Boom 🤛"},
    {"Emoji": "🤜", "Name": "Right-Facing Fist", "Category": "Hand", "Example": "Boom 🤜"},
    {"Emoji": "💪", "Name": "Flexed Biceps", "Category": "Arm", "Example": "Stay strong 💪"},
    {"Emoji": "🦵", "Name": "Leg", "Category": "Leg", "Example": "Running fast 🦵"},
    {"Emoji": "🦶", "Name": "Foot", "Category": "Leg", "Example": "Stepping forward 🦶"},
    {"Emoji": "🤲", "Name": "Palms Up Together", "Category": "Hand", "Example": "Please help 🤲"},
    {"Emoji": "🤞", "Name": "Crossed Fingers", "Category": "Hand", "Example": "Wish me luck 🤞"},
    {"Emoji": "🖕", "Name": "Middle Finger", "Category": "Hand", "Example": "Don’t mess with me 🖕"},
    {"Emoji": "👉", "Name": "Pointing Right", "Category": "Hand", "Example": "Look at this 👉"},
    {"Emoji": "👈", "Name": "Pointing Left", "Category": "Hand", "Example": "Check that out 👈"},
    {"Emoji": "👆", "Name": "Pointing Up", "Category": "Hand", "Example": "See above 👆"},
    {"Emoji": "👇", "Name": "Pointing Down", "Category": "Hand", "Example": "See below 👇"},
]

# Step 3: Create DataFrame & Analyze Sentiment
df = pd.DataFrame(body_emojis)
df["Polarity"] = df["Example"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Step 4: Display Table
print("📌 Hand, Arm & Leg Emoji Sentiment Table")
display(df[["Emoji", "Name", "Category", "Example", "Polarity"]])

# Step 5: Category Count Plot
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Category", order=df["Category"].value_counts().index, palette="Blues")
plt.title("🧠 Emoji Category Distribution", fontsize=16)
plt.ylabel("Count")
plt.xlabel("Body Part")
plt.tight_layout()
plt.show()

# Step 6: Sentiment Polarity Plot
plt.figure(figsize=(16,8))
sns.barplot(data=df, x="Emoji", y="Polarity", hue="Category", dodge=False, palette="coolwarm", edgecolor="black")
plt.title("📊 Sentiment Polarity of Hand, Arm & Leg Emojis", fontsize=16)
plt.ylabel("Polarity Score (-1 to 1)")
plt.xlabel("Emoji")
plt.ylim(-1, 1)
plt.axhline(0, color="gray", linestyle="--")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
