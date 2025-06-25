import os
from flask import Flask, request, jsonify
from microMLP import MLP

# Chargement d’un mini modèle (exemple, à adapter)
model = MLP()
model.load("model.json")  # Faut que t’aies ce fichier avec un modèle entraîné

app = Flask(__name__)

@app.route('/ask')
def ask():
    q = request.args.get('q', '')
    if not q:
        return jsonify({"error": "Pas de question donnée"}), 400

    # Ici tu convertis la question en vecteurs (à adapter selon ton modèle)
    input_data = [len(q)]  # Juste un exemple débile : la longueur du texte
    output = model.predict(input_data)

    return jsonify({"réponse": str(output)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
