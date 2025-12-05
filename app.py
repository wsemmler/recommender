
import flask
import transformers
import flask_cors
import sentence_transformers
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app) 
model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_books(books, top_n=3):
    texts = [b['title'] + ' ' + b['categories'] for b in books]
    embeddings = model.encode(texts, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)

    used_recommendations = set()
    recommendations = []

    for idx, book in enumerate(books):
        cos_sim[idx, idx] = -1.0  # exclude itself
        sorted_indices = torch.argsort(cos_sim[idx], descending=True)
        chosen_title = None       
        for i in sorted_indices:
            title = books[int(i)]['title']
            if title not in used_recommendations:
                chosen_title = title
                used_recommendations.add(title)
                break
        if chosen_title is None:
            chosen_title = books[int(sorted_indices[0])]['title']

        recommendations.append({
            'book': book['title'],
            'recommendations': [chosen_title]   # only one result!
        })

    return recommendations

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    books = data.get('books', [])
    recs = recommend_books(books)
    return jsonify(recs)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)