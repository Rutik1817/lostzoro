from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance

app = Flask(__name__)

# Read the anime data
data = pd.read_csv('ANM2.csv')

# Filter out Hentai genre
data = data[data['genre'] != 'Hentai']

# Extract genre information and preprocess
genre = data['genre'].values
lis = []
for i in genre:
    i = str(i)
    for p in i.split(','):
        if p not in lis:
            lis.append(p)

# Create feature vectors for each anime
dic = {}
t = []
for i in genre:
    for j in lis:
        dic[j] = 0
    i = str(i)
    for p in i.split(','):
        dic[p] += 1
    t.append(list(dic.values()))

X = np.array(t)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET','POST'])
def recommend():
    anime_name = request.form['anime_name'].lower()
    num_recommendations = int(request.form['num_recommendations'])
    h = -1
    for i in range(len(data)):
        if anime_name in data['name'][i].lower():
            h = i
            break
    imp = []
    if h == -1:
        return render_template('recommendations.html', recommendations=[], not_found=True)
    else:
        for i in range(len(t)):
            if i == h:
                continue
            else:
                if len(imp) < num_recommendations:
                    imp.append([distance.euclidean(t[i], t[h]), t[i], i])
                else:
                    imp.sort()
                    if imp[num_recommendations - 1][0] > distance.euclidean(t[i], t[h]):
                        del imp[num_recommendations - 1]
                        imp.append([distance.euclidean(t[i], t[h]), t[i], i])
        recommendations = []
        for i in imp:
            recommendations.append(data['name'].values[i[2]])
        return render_template('recommendations.html', recommendations=recommendations, not_found=False)

if __name__ == '__main__':
    app.run(debug=True)
