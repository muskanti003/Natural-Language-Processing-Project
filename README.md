# Natural-Language-Processing-Project
Mood Muse AI: Intelligent Music &amp; Motivation Companion 
  # 🎵 Emotion-Based Music Recommender (Colab/Jupyter Safe Version)

!pip install transformers torch sentencepiece gtts emoji pandas matplotlib > /dev/null

import torch
from transformers import pipeline
import pandas as pd
from IPython.display import display, HTML, clear_output
import time, emoji
from google.colab import files
import IPython.display as ipd
from gtts import gTTS
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# Step 1: Load pretrained BERT emotion classifier
print("🔄 Loading BERT emotion model...")
emotion_model = pipeline("text-classification",
                         model="bhadresh-savani/distilbert-base-uncased-emotion")
print("✅ Model ready!\n")

# Step 2: Hindi song database (7 per emotion)
songs = {
    "happy": [
        ("Jeene Laga Hoon", "https://youtu.be/vP_lgaVhb1Y?si=Ol0nvQidvKiebYXY"),
        ("Gallan Goodiyan", "https://youtu.be/jCEdTq3j-0U?si=ocSsqzifuC1R-ZGb"),
        ("Badtameez Dil", "https://youtu.be/II2EO3Nw4m0?si=cbECXfRY1NfNnM9i"),
        ("Ude Dil Befikre", "https://youtu.be/gXe-KWe-YMs?si=AtwS83lSQpSj8hAL"),
        ("Dil Dhadakne Do", "https://youtu.be/WuMWwPHTSoY?si=fE7zRbxj8i7FULpV"),
        ("Matargasti", "https://youtu.be/6vKucgAeF_Q?si=iudYo4db5AX5sjrE"),
        ("Tum hi bandu", "https://youtu.be/DzWP07XqoeQ?si=qgxh4V3CdiWvfqYK")
    ],
    "sad": [
        ("Channa Mereya", "https://youtu.be/bzSTpdcs-EI?si=l902b3zD-XfAsUzG"),
        ("Agar Tum Saath Ho", "https://youtu.be/sK7riqg2mr4?si=f3EaM7jYMk-A6stx"),
        ("Tujhe Kitna Chahne Lage", "https://youtu.be/2IGDsD-dLF8?si=p4G0qN5-Y1Y8MJfM"),
        ("Phir Le Aaya Dil", "https://youtu.be/k6BnSIs3XUQ?si=MEuGmjWzymQzc6ru"),
        ("Kal Ho Naa Ho", "https://youtu.be/g0eO74UmRBs?si=Dz8sD7qRQYyeliuT"),
        ("Tum hi Aana", "https://youtu.be/RemShT6JAHw?si=uf9GpLGkM2EIvmrD"),
        ("Humari Adhuri Kahani", "https://youtu.be/sVRwZEkXepg?si=mQqWeyRseVY67ZxA")
    ],
    "angry": [
        ("Zinda", "https://youtu.be/Ax0G_P2dSBw?si=lsnE_iD4epodIXnW"),
        ("Sultan Title Track", "https://youtu.be/abiL84EAWSY?si=2f6A4HvzlIoQLpM6"),
        ("Brothers Anthem", "https://youtu.be/IjBAgWKW12Y?si=wea73Vel46CM4veW"),
        ("Lakshya", "https://youtu.be/8DMF0U6xV78?si=m4iHsFVX9bD2vchd"),
        ("Chak De India", "https://youtu.be/bnqLzCsffwY?si=BnwQMdwxNMSk81R-"),
        ("Jee Karda", "https://youtu.be/VAJK04HOLd0?si=lIrnHoJOVqUEImHb"),
        ("Ziddi Dil", "https://youtu.be/puKD3nkB1h4?si=2jhdRkJdOX8boFSG")
    ],
    "fear": [
        ("Khaabon Ke Parinday", "https://youtu.be/R0XjwtP_iTY?si=NkMykL8oBBbIezjw"),
        ("Raat Bhar", "https://youtu.be/AYcxiROIktI?si=hPvo2bKB82unWd6w"),
        ("Naina Da Kya Kasoor", "https://youtu.be/WfDFWLZgQwE?si=fCRtTHlGlCG74vti"),
        ("Dil Ibaadat", "https://youtu.be/U2QNhsAgIIE?si=oQLd2jskHLIGAG6k"),
        ("Tera Ban Jaunga", "https://youtu.be/Qdz5n1Xe5Qo?si=rg1MkzGwEI8nXguz"),
        ("Phir le aaya dil","https://youtu.be/Z93rAu25KqI?si=JwhGePWX9wWg42gl"),
        ("Hai apna dil", "https://youtu.be/adDtRrqJzys?si=JXWCa5NdgXRkFoi0")
    ],
    "surprise": [
        ("Senorita", "https://youtu.be/2Z0Put0teCM?si=IDNOw_9-SenTNGzg"),
        ("Malhari", "https://youtu.be/l_MyUGq7pgs?si=IrkXZr_hEUiZXVyi"),
        ("Subha Hone Na De", "https://youtu.be/iWhZ62o-arQ?si=LWgC1ThTOhbso-Zf"),
        ("Tune Maari Entriyaan", "https://youtu.be/2I3NgxDAiqE?si=t9Ojhg2ACIQ4guQl"),
        ("Koi Kahe Kehta Rahe", "https://youtu.be/ctJI7pCbxAo?si=dITs-iqownKIhGeQ"),
        ("Uyi Amma", "https://youtu.be/FZLadzn5i6Q?si=iRcFpUHzlYKWf9lI"),
        ("Gumshuda", "https://youtu.be/q5sDt9Mmgb0?si=2GKphW9zmtZM5m48")
    ],
    "love": [
        ("Tum Se Hi", "https://youtu.be/Cb6wuzOurPc?si=2DXQVJPz0Y3i5S8T"),
        ("Raabta", "https://youtu.be/zlt38OOqwDc?si=N_rXUhg0g3uk0yl1"),
        ("Pee Loon", "https://youtu.be/WCTro3qabjE?si=xeSl8KlgG8UaaNFh"),
        ("Tera Hone Laga Hoon", "https://youtu.be/rTuxUAuJRyY?si=wtrK7RB7xhwic1eo"),
        ("Hawayein", "https://youtu.be/cs1e0fRyI18?si=9aCXwX6utoUwScqX"),
        ("Raatan Lambiyaan", "https://youtu.be/orYf6VDtj_k?si=WEcBMudm514_u9Sn"),
        ("Tum Tak", "https://youtu.be/1nWQs6IxTrY?si=Z7mxey5iFby7vDvw"),
    ],
    "calm": [
        ("Ilahi", "https://youtu.be/wM4xpaWV5s4?si=yCdp_7vs2Ks17qjT"),
        ("Kun Faya Kun", "https://youtu.be/T94PHkuydcw?si=a0_hMu6AMxvYxEF_"),
        ("Manwa Laage", "https://youtu.youtube/d8IT-16kA8M?si=LZycJXLR68MDDAY6"),
        ("Khuda Jaane", "https://youtu.be/cmMiyZaSELo?si=EDlM1kqvVStar5Ss"),
        ("Tera Yaar Hoon Main", "https://youtu.be/EatzcaVJRMs?si=N_OLARFhdLzOj9M6"),
        ("Ishq Bulaava", "https://youtu.be/Oo5tqEWm-jM?si=zACpAJCTfr4L7pQY"),
        ("Ek din aap yun", "https://youtu.be/1sRaLqtHXQU?si=jBtMiY796tcYJEaj")
    ],
    "energetic": [
        ("Apna Time Aayega", "https://youtu.be/jFGKJBPFdUA?si=Nc9S9mWIBJZdwuCl"),
        ("Born to Shine", "https://youtu.be/dCmp56tSSmA?si=E2bqo40V8Ng-uUHX"),
        ("Kar Har Maidaan Fateh", "https://youtu.be/RCgbE6eS-DU?si=ZB6XY_mS_F-NN146"),
        ("Jai Jai Shivshankar", "https://youtu.be/Mpy0UzbHNr0?si=TD1LcfRhrxcTEdtX"),
        ("Malang Title Track", "https://youtu.be/SxoTAvwCr4A?si=gg8wPM-Xrsb7S8HK"),
        ("Afghan Jalebi", "https://youtu.be/zC3UbTf4qrM?si=T8QgAto0uDD3QHdC"),
        ("Sooraj Dooba Hain", "https://youtu.be/nJZcbidTutE?si=PDE2mh_fM6BjLOpw")
    ],
    "neutral": [
        ("Safarnama", "https://youtu.be/sOhESxhibAM?si=1vOL_FKadnS8kikk"),
        ("Humnava Mere", "https://youtu.be/TmRgK-pXH9c?si=j_XafcZ8IBf8u1JH"),
        ("Tera Zikr", "https://youtu.be/eK0IIyBlYew?si=CctjJyB6V86IN_Th"),
        ("Phir Se Ud Chala", "https://youtu.be/2mWaqsC3U7k?si=HVgRjmKavL3WyTVg"),
        ("Zinda Rehti Hain Mohabbatein", "https://youtu.be/bg64DpNPZZU?si=svHr1xMZeDnX8Rin"),
        ("Zindgi Kuch toh Bata", "https://youtu.be/ITyHqStTDeg?si=xaGV7-4m-bNJDoem"),
        ("Dekha Ek Khwab", "https://youtu.be/7dO_MS9tZ5E?si=4Bzic1GxKEq4w6An")
    ]
}

# Step 3: Animation helper
def animate_emoji(e):
    for _ in range(5):
        clear_output(wait=True)
        print(emoji.emojize(f"{e*5} Analyzing your mood..."))
        time.sleep(0.3)

# Step 4: Detect emotion
def detect_emotion(text):
    result = emotion_model(
        text,
        truncation=True,
        max_length=512,
        return_all_scores=False
    )[0]
    label = result['label'].lower()
    print(f"\n🧠 Emotion Detected: {label.upper()}")
    return label

#✅ Step 4A: Add LSTM model (for text emotion)
texts = ["I am happy", "Feeling joyful", "I am sad", "I am angry", "I am scared", "I am surprised", "I love you", "Feeling calm"]
labels = [0, 0, 1, 2, 3, 4, 5, 6]
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=10)
y = np.array(labels)

lstm_model = Sequential([
    Embedding(1000, 32, input_length=10),
    LSTM(32),
    Dropout(0.3),
    Dense(7, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X, y, epochs=10, verbose=0)

def detect_lstm_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    seq_pad = pad_sequences(seq, maxlen=10)
    pred = np.argmax(lstm_model.predict(seq_pad), axis=1)[0]
    emotion_list = ["happy", "sad", "angry", "fear", "surprise", "love", "calm"]
    return emotion_list[pred]

# ✅ Step 4B: Detect emotion + visualize
def detect_emotion(text):
    results = emotion_model(text, truncation=True, max_length=512, return_all_scores=True)[0]
    df = pd.DataFrame(results).sort_values(by="score", ascending=True)
    plt.figure(figsize=(7,4))
    plt.barh(df["label"], df["score"], color="#4CAF50")
    plt.xlabel("Confidence")
    plt.title("BERT Emotion Confidence Visualization")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.show()
    label = max(results, key=lambda x: x["score"])["label"].lower()
    print(f"\n🧠 LSTM Emotion Detected: {label.upper()}")
    return label

# Step 5: Show 3 random songs
def show_songs(emotion):
    emoji_map = {
        "fear": "😨",
        "surprise": "😲",
        "happy": "😊",
        "sad": "😢",
        "angry": "😡",
        "love": "😍",
        "calm": "😌",
        "energetic": "😎",
        "neutral": "🤔"
    }
    if emotion not in songs:
        print("😅 No songs for this emotion.")
        return
    icon = emoji_map.get(emotion, "🎵")
    print(f"\n{icon} Songs for your mood ({emotion.upper()}):\n")
    sample = random.sample(songs[emotion], 3)
    df = pd.DataFrame(sample, columns=["Song", "YouTube"])
    df["YouTube"] = df["YouTube"].apply(lambda x: f'<a href="{x}" target="_blank">🎵 Listen</a>')
    display(HTML(df.to_html(escape=False, index=False)))

# Step 6: Voice upload
def get_voice_text():
    print("🎤 Upload a short .wav file (≤10s):")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    ipd.display(ipd.Audio(filename))
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    out = whisper(filename)
    text = out["text"]
    print(f"\n🗣️ You said: {text}")
    return text

# Step 7: Run
choice = input("🎙️ Upload voice (v) or type (t)? ").lower().strip()
if "v" in choice:
    text = get_voice_text()
else:
    text = input("💬 Type your sentence: ")

animate_emoji("🎶")
emotion = detect_emotion(text)

emotion_map = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "fear",
    "love": "love",
    "surprise": "surprise",
    "energetic": "energetic"
}
emotion = emotion_map.get(emotion, emotion)

show_songs(emotion)

tts = gTTS(f"You seem {emotion} today. Here are some Hindi songs for you!", lang="en")
tts.save("emotion.mp3")
ipd.display(ipd.Audio("emotion.mp3", autoplay=True))

print("\n🎧 Enjoy your emotion-matching Hindi songs! 🇮🇳🎶")

# 🎨 Display Beautiful HTML with Floating Emojis
quote = f"Music always matches your {emotion} emotions 🎶"
html_code = f"""
<style>
@keyframes float {{
  0% {{ transform: translateY(0) rotate(0deg); opacity: 1; }}
  50% {{ transform: translateY(-60px) rotate(20deg); opacity: 0.8; }}
  100% {{ transform: translateY(0) rotate(0deg); opacity: 1; }}
}}
.floating-emoji {{
  position: absolute;
  animation: float 5s ease-in-out infinite;
  font-size: 35px;
}}
.floating-emoji:nth-child(1) {{ top: 10%; left: 10%; animation-delay: 0s; }}
.floating-emoji:nth-child(2) {{ top: 20%; left: 80%; animation-delay: 1s; }}
.floating-emoji:nth-child(3) {{ top: 70%; left: 15%; animation-delay: 2s; }}
.floating-emoji:nth-child(4) {{ top: 40%; left: 60%; animation-delay: 3s; }}
.floating-emoji:nth-child(5) {{ top: 80%; left: 50%; animation-delay: 4s; }}
.floating-emoji:nth-child(6) {{ top: 30%; left: 30%; animation-delay: 1.5s; }}
.floating-emoji:nth-child(7) {{ top: 60%; left: 75%; animation-delay: 2.5s; }}
.floating-emoji:nth-child(8) {{ top: 50%; left: 45%; animation-delay: 3.5s; }}
</style>

<div style='position: relative; overflow: hidden;
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 25%, #f6d365 50%, #cfd9df 75%, #e2ebf0 100%);
            border-radius: 20px; padding: 25px; text-align: center;
            font-family: "Poppins", sans-serif; box-shadow: 0 0 25px rgba(0,0,0,0.2);
            color: #003366;'>

  <div class='floating-emoji'>🎵</div>
  <div class='floating-emoji'>🎧</div>
  <div class='floating-emoji'>🎶</div>
  <div class='floating-emoji'>💫</div>
  <div class='floating-emoji'>🌈</div>
  <div class='floating-emoji'>🎼</div>
  <div class='floating-emoji'>💖</div>
  <div class='floating-emoji'>🔥</div>
  <div class='floating-emoji'>😘</div>
  <div class='floating-emoji'>🫣</div>
  <div class='floating-emoji'>😌</div>
  <h2 style='color:#004d40; font-size:32px;'>
    🎭 Detected Emotion: <span style='color:#006400;'>{emotion.upper()}</span>
  </h2>

  <p style='font-size:20px; font-style:italic; background-color:#d9f7f7;
            display:inline-block; padding:10px 20px; border-radius:25px;
            color:#004d40;'>{quote}</p>

  <h3 style='margin-top:25px; color:#003366;'>🎶 Songs for your mood:</h3>
  <table style='margin:auto; border-collapse:collapse; width:75%; font-size:18px;'>
    <tr style='background:#004d40; color:white;'>
      <th>🎵 Song</th><th>🔗 YouTube</th>
    </tr>
"""

for s, link in songs.get(emotion, []):
    html_code += f"<tr style='background:#f5f5f5; color:#003366;'><td>{s}</td><td><a href='{link}' target='_blank' style='color:#006400; text-decoration:none;'>🎧 Listen</a></td></tr>"

html_code += """
  </table>
  <p style='margin-top:25px; font-weight:bold; font-size:18px; color:#004d40;'>
  🌈 Enjoy your personalized musical therapy session! 🎧💫
  </p>
</div>
"""
display(HTML(html_code))

# 💡 Deep Learning + NLP Enhancements
print("\n🤖 Adding deep learning intelligence...")

# Sentiment + Emotion Dual Analysis
from transformers import pipeline as pipe
sentiment_model = pipe("sentiment-analysis")
sentiment_result = sentiment_model(text)[0]['label']
emotion_result = emotion_model(text)[0]['label']

dual_html = f"""
<div style='background:linear-gradient(135deg,#d4fc79 0%,#96e6a1 100%);
            border-radius:15px;padding:20px;margin-top:20px;text-align:center;
            font-family:Poppins,sans-serif;color:#003366;box-shadow:0 0 15px rgba(0,0,0,0.2);'>
 <h2 style="color:#001f3f;">🧠 Dual NLP Analysis</h2>
  <p style='font-size:18px;'>Your text shows <b>{sentiment_result.lower()}</b> sentiment and expresses <b>{emotion_result.lower()}</b> emotion.</p>
  <p style='font-size:16px;color:#004d40;'>Two transformer models used: one for <b>sentiment polarity</b> and another for <b>emotional tone</b>.</p>
</div>
"""
display(HTML(dual_html))

# AI Quote Generation
from transformers import pipeline as pl
quote_gen = pl("text-generation", model="gpt2")
prompt = f"Write a short inspiring quote for someone feeling {emotion}:"
quote_output = quote_gen(prompt, max_length=25, num_return_sequences=1)[0]['generated_text']

quote_html = f"""
<div style='background:linear-gradient(135deg,#fddb92 0%,#d1fdff 100%);
            border-radius:15px;padding:20px;margin-top:20px;text-align:center;
            font-family:Poppins,sans-serif;color:#003366;box-shadow:0 0 15px rgba(0,0,0,0.2);'>
  <h2 style="color:#001f3f;">💡 AI-Generated emotion based inspiring quote for You</h2>
  <p style='font-style:italic;font-size:18px;'>{quote_output}</p>
  <p style='font-size:15px;color:#555;'>Generated using GPT-2 (Transformer-based language model)</p>
</div>
"""
display(HTML(quote_html))
