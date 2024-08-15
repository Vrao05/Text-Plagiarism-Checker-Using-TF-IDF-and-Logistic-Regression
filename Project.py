
from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PlagiarismDetector:
    def __init__(self):
        self.texts = []
        self.filenames = []
        self.vectorizer = TfidfVectorizer()

    def add_file(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                self.texts.append(content)
                self.filenames.append(os.path.basename(filename))
                print(f"Added: {os.path.basename(filename)}")
        else:
            print(f"File not found: {filename}")

    def create_tfidf_vectors(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def find_plagiarism(self, threshold=0.5):
        similarities = cosine_similarity(self.tfidf_matrix)
        results = []

        for i in range(len(self.filenames)):
            for j in range(i + 1, len(self.filenames)):
                similarity = similarities[i, j]
                if similarity > threshold:
                    results.append({
                        'file1': self.filenames[i],
                        'file2': self.filenames[j],
                        'similarity': similarity * 100
                    })

        return results


app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = '/content/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt'}

detector = PlagiarismDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Plagiarism Detector</title>
    </head>
    <body>
        <h1>Upload Files for Plagiarism Detection</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="files[]" multiple>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return 'No file part', 400

    files = request.files.getlist('files[]')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            detector.add_file(filepath)
        else:
            return f'File {file.filename} is not allowed.', 400

    detector.create_tfidf_vectors()
    return redirect(url_for('result'))

@app.route('/result')
def result():
    plagiarized_files = detector.find_plagiarism()
    html_content = '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Plagiarism Detection Results</title>
    </head>
    <body>
        <h1>Plagiarism Detection Results</h1>
    '''

    if plagiarized_files:
        html_content += '<p>The following files have been detected as plagiarized:</p><ul>'
        for result in plagiarized_files:
            html_content += f'<li>{result["file1"]} is plagiarized with {result["file2"]} with a similarity of {result["similarity"]:.2f}%</li>'
        html_content += '</ul>'
    else:
        html_content += '<p>No plagiarized files detected.</p>'

    html_content += '''
        <a href="/">Go back to upload more files</a>
    </body>
    </html>
    '''

    return html_content
public_url = ngrok.connect(5000)
print(f" * Ngrok tunnel available at: {public_url}")
app.run(port=5000)
