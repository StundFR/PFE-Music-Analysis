import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from textwrap import wrap
from tqdm import tqdm

import re
from langdetect import detect
import spacy
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from bs4 import BeautifulSoup
from requests import get
import json
import wikipedia
import lyricsgenius as genius
import spotipy
from spotipy.oauth2 import SpotifyOAuth

###############################################################################################################################
############################################################ 0 ##########################################################
###############################################################################################################################

def printFull(data, nb_rows=None, nb_col=10):
    pd.set_option('display.max_rows', nb_rows)
    pd.set_option('display.max_columns', nb_col)
    print(data)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

###############################################################################################################################
############################################################ ETAPE  ##########################################################
###############################################################################################################################

###############################################################################################################################
############################################################ ETAPE 2 ##########################################################
###############################################################################################################################

def find_title_in_wikipedia(title, pourcentage=0.3):
    words = ["(chanteur)", "(chanteuse)", "(groupe)", "(rappeur)", "(rappeuse)", "(musicien)", "(chanteur français)", "(france)", "(producteur)", "(artiste)", "(groupe de musique)"]

    wikipedia.set_lang("fr")
    results = wikipedia.search(title, results=10)
    distance = []
    if len(results) > 0:
        for element in results:
            if any((w in element.lower()) and (edit_distance(element.lower().split(" (")[0].strip(), title.lower().strip())/len(title) < pourcentage) for w in words):
                return element

            distance.append(edit_distance(title.lower().strip(), element.lower().strip()))

        return results[np.argmin(distance)] if min(distance)/len(title) < pourcentage else np.NaN


def get_summary_from_wikipedia(title):
    url = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "",
        "explaintext": "",
        "titles": title,
        "redirects": "",
    }

    response = get(url=url, params=params)
    data = json.loads(response.text)
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"].split("\n")[0]

def wiki_birth(title):
    cols = ["Naissance", "Pays d'origine", "Origine", "Nationalité", "Pays", "Summary"]
    dic = {w : np.NaN for w in cols}

    if title is None or title is np.NaN:
        return dic

    url = f"https://fr.wikipedia.org/wiki/{title}"
    rq = get(url)

    if not rq.ok:
        return dic
    
    soup = BeautifulSoup(rq.text, features="html.parser")
    tables = soup.findAll("table")

    for table in tables:
        trs = table.findAll("tr")

        for tr in trs:
            th = tr.find("th")

            if th is not None:
                for w in cols:
                    if w in th.text:
                        td = tr.find("td")
                        if td is not None:
                            dic[w] = td.text.strip().lower()

    # Garde la première nationalité en cas de double nationalité
    if not dic["Nationalité"] is np.NaN:
        dic["Nationalité"] = dic["Nationalité"].split(" ")[0]

    summary = get_summary_from_wikipedia(title)
    dic["Summary"] = summary.lower().strip()

    return dic


def add_id(df : pd.DataFrame, cols : list, id_name : str):
    df_unique = df[cols].drop_duplicates(ignore_index=True)
    df_unique[id_name] = df_unique.index

    return df.merge(df_unique, on=cols, how="inner")

###############################################################################################################################
##################################################### SCRAPPING ARTIST DATA ###################################################
###############################################################################################################################
#On recalcul les distances entre les noms d'artistes et les pages wikipédia pour vérifier
def calc_dist(artist, wiki):
    if wiki == "manuel":
        return 0

    words = ["(chanteur)", "(chanteuse)", "(groupe)", "(rappeur)", "(rappeuse)", "(musicien)", "(chanteur français)", "(france)", "(producteur)", "(artiste)", "(groupe de musique)"]

    if any(w in wiki.lower() for w in words):
        dist = edit_distance(artist.lower().strip(), wiki.lower().split(" (")[0].strip())
        if dist/len(artist) < 0.4:
            return dist
    
    return edit_distance(artist.lower().strip(), wiki.lower().strip())

def pie_chart(df, col, title, na=True, legends=[], colors=["green", "red"], figsize=(3,3)):
    if na:
        df[col].isna().value_counts(normalize=True).round(2).plot.pie(autopct="%.2f%%", figsize=figsize, colors=colors)
    else:
        df[col].value_counts(normalize=True).round(2).plot.pie(autopct="%.2f%%", figsize=figsize, colors=colors)
    plt.title(title, color='white')
    plt.gcf().set_facecolor('black')
    plt.legend(legends, bbox_to_anchor=(1, 0.5))
    plt.show()

def category_count(cols, df, figsize=(40,7), top=10):
    length = []
    for col in cols:
        length.append(len(df[col].unique()))

    fig, ax = plt.subplots(1, len(cols)+1, figsize=figsize)

    sns.barplot(x=cols, y=length, ax=ax[0])
    ax[0].set_title("Nombre de valeurs différentes pour chaque colonne")

    for axi, col in zip(ax[1:], cols):
        df[col].value_counts().head(top)[::-1].plot.barh(ax=axi)
        axi.set(title=f"Top {top} de {col}", xlabel="Nombre d'occurence")
        axi.set_yticklabels(["\n".join(wrap(elem.get_text(), 10)) for elem in axi.get_yticklabels()])

###############################################################################################################################
############################################################ ETAPE 4 ##########################################################
###############################################################################################################################

def extraire_date(naissance):
    if naissance is np.nan:
        return np.nan, np.nan, np.nan
    pattern = r"(\d{1,2})\s+(\w+)\s+(\d{4})"
    match = re.search(pattern, naissance)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return np.nan, np.nan, np.nan

def get_nationality(data_to_check, check_list):
    if data_to_check is np.NaN:
        return np.NaN
        
    for r in check_list:
        regex = r"([\d)()\], ]|^)"+ r.lower() + r"([.,\[) ]|$)"

        if not re.search(regex, data_to_check.lower()) is None:
            return r.lower()
    return np.NaN

#Cela va permettre de normaliser les noms des pays et des nationalites
def cleanning(data, replace_words, p=0.3):
    if data is np.NaN:
        return np.NaN
    
    distances = []
    for r in replace_words:
        distances.append(edit_distance(data.lower(), r.lower())/len(data))
    return replace_words[np.argmin(distances)] if np.min(distances) < p else np.NaN

#Permet de récupérer la localisation d'un artiste à partir de sa naissance ou sommaire wikipedia
def get_localisation(row, localisation):
    for r in localisation:
        regex = r"([\d)()\] ]|^)" + r.lower() + r"([.,\[) ]|$)"

        if (not re.search(regex, str(row["summary"]).lower()) is None) or (not re.search(regex, str(row["naissance"]).lower()) is None):
            return r.lower()
    return np.NaN

###############################################################################################################################
############################################################ ETAPE 5 ##########################################################
###############################################################################################################################

def find_lyrics(x):
    g = genius.Genius("19WnPxjd0b-zuJeynlxyOZ7UItqsAPQbVk3libvr4DsUGLgzyNjwp4jig6dMerlq")
    g.verbose = False
    try:
        song = g.search_song(title=x["music"], artist=x["artist"], get_full_info=False)
        return song.lyrics
    except:
        return np.NaN

###############################################################################################################################
############################################################ ETAPE 6 ##########################################################
###############################################################################################################################

def remove_translation(lyrics:str):
    start = lyrics.split(" Lyrics")[0]
    isUpperNum = [k for k, l in enumerate(start) if l.isupper() or l.isnumeric()]
    idx = isUpperNum[-1]

    k = 2
    while lyrics[idx-1] in ["-", " ", ".", "(", "&", "/"] or lyrics[idx-1].isupper() or lyrics[idx-1].isnumeric():
        idx = isUpperNum[-k]
        k += 1
        
    return lyrics[idx:]

def check_good_lyrics_without_translation(lyrics, title, p=0.3):
    if lyrics is np.NaN:
        return False
    
    song_title_lyrics = lyrics.lower().split(" lyrics")[0]
    song_title_lyrics_without_parent = re.sub(r" [\(\[].*?[\)\]]", "", song_title_lyrics)
    distance_without = edit_distance(song_title_lyrics_without_parent, title.lower())
    distance = edit_distance(song_title_lyrics, title.lower())

    return distance/len(title) <= p or distance_without/len(title) <= p


def check_good_lyrics_with_translation(lyrics, title, p=0.3):
    if lyrics is np.NaN:
        return False

    song_title_lyrics = lyrics.lower().split(" lyrics")[0]
    song_title_lyrics = song_title_lyrics[-len(title):]
    distance = edit_distance(song_title_lyrics, title.lower())
    return distance/len(title) <= p


def remove_title_lyrics(lyrics:str):
    idx = re.search(r" lyrics", lyrics.lower()).span()[1]
    return lyrics[idx:]


def remove_crochet(lyrics:str):
    lyrics = re.sub(r"[\(\[].*?[\)\]]", "", lyrics)
    lyrics = re.sub(r"\n", " ", lyrics)
    lyrics = re.sub(" +", " ", lyrics)
    return lyrics


def cleanning_lyrics(row):
    lyrics = row["lyrics"]
    title = row["music"]

    if lyrics is np.NaN:
        return np.NaN

    if "lyrics" in title:
        raise ValueError("Y'a le mot lyrics dans le titre")
        
    # On s'occupe des paroles qui ont des traductions
    if lyrics.lower().startswith("translation"):
        if not check_good_lyrics_with_translation(lyrics, title) and not check_good_lyrics_without_translation(remove_translation(lyrics), title):
            return np.NaN
        else:
            lyrics = remove_translation(lyrics)
    else:
        if not check_good_lyrics_without_translation(lyrics, title):
            return np.NaN

    # On met tous en minuscule
    lyrics = lyrics.lower()

    # On enleve le titre du debut
    lyrics = remove_title_lyrics(lyrics)
        
    # On enleve les crochets
    lyrics = remove_crochet(lyrics)

    # On enleve le 'embed' a la fin
    if lyrics.endswith("embed"):
        lyrics = lyrics[:-5]

    lyrics = lyrics.replace("you might also like", "")

    lyrics = lyrics.strip()

    return lyrics

###############################################################################################################################
############################################################ ETAPE 8 ##########################################################
###############################################################################################################################

def get_features(row):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='69c6ef763f7c4b2981255ad74d81e75d', client_secret='8b36a262fb2f4c7c9e6051924d1986f6', redirect_uri='http://localhost:8888/callback', scope=['app-remote-control']))

    keys = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]

    title = row["music"]
    artist = row["artist"]
    
    song = sp.search(q='artist:' + artist + ' track:' + title, type='track', limit=1)

    if (not song is None) and (song['tracks']['items']):
        features = sp.audio_features(song['tracks']['items'][0]["id"])[0]

        if not features is None:
            return [features[key] for key in keys]
            
    return [np.nan]*len(keys)

###############################################################################################################################
############################################################ ETAPE 8 ##########################################################
###############################################################################################################################

def lematization(lyrics : str, nlp : spacy.lang):
    lyrics_lemma = [token.lemma_ for token in nlp(lyrics) if not token.is_stop]
    return " ".join(lyrics_lemma)

def tfidf_mat(corpus : list):
    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform(corpus)
    return pd.DataFrame(mat.toarray(), columns=tfidf.get_feature_names_out())

# Lyrics concatenation by a column
def lyrics_by(data, by, nlp):
    lyrics = data.groupby(by)["lyrics"].apply(lambda x : " ".join(x.to_list()))
    lyrics = lyrics.apply(lambda x : lematization(x, nlp))
    return lyrics

def tfidf_by(data, by, nlp):
    lyrics = lyrics_by(data, by, nlp)
    index = lyrics.index.to_list()
    tfidf = tfidf_mat(lyrics)
    tfidf.index = index
    return tfidf

###############################################################################################################################
############################################################ ETAPE 9 ##########################################################
###############################################################################################################################

def make_bag_of_words(lyrics:str, nlp, stopwords = None):
    if lyrics is np.NaN:
        return np.NaN

    if stopwords is None:
        stopwords = nlp.Defaults.stop_words

    bow = {}
    for token in nlp(lyrics):
        if not str(token.lemma_) in stopwords:
            if bow.get(token.lemma_) is None:
                bow[token.lemma_] = 1
            else:
                bow[token.lemma_] += 1
    
    return bow


def compare_words(df : pd.DataFrame, col : str):
    series = []
    iterate = df[col].unique().tolist()
    if np.NaN in iterate:
        iterate.remove(np.NaN)

    for i in iterate:
        s = pd.DataFrame(df[df[col] == i].groupby("mot")["nb"].sum())
        series.append(s)

    df = pd.concat(series, axis=1).fillna(0)
    df.columns = iterate
    return df

def plot_compare_words(df, nb_rows, nb_cols, figsize=(30,10)):
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=figsize)

    for col, ax in zip(sorted(df.columns), ax.flatten()):
        df[col].sort_values()[-20:].plot.barh(ax=ax)
        ax.set(title=f"{col}", xlabel="Nb", ylabel="Mots")

###############################################################################################################################
############################################################ POWER BI ##########################################################
###############################################################################################################################

def homogeneous_data(data_to_homogeneous, base_data_df):
    data = base_data_df.dropna().unique().tolist()

    dist = []
    for d in data:
        dist.append(edit_distance(data_to_homogeneous, d))

    return data[np.argmin(dist)]