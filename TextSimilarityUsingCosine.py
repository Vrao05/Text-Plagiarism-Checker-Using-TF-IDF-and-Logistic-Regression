from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stopwords = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    ])

    tokens = [word for word in tokens if word not in stopwords]

    return ' '.join(tokens)

def compute_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity
text1 = "Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions, in addition to the study of other forms of nuclear matter.Nuclear physics should not be confused with atomic physics, which studies the atom as a whole, including its electrons.Discoveries in nuclear physics have led to applications in many fields. This includes nuclear power, nuclear weapons, nuclear medicine and magnetic resonance imaging, industrial and agricultural isotopes, ion implantation in materials engineering, and radiocarbon dating in geology and archaeology. Such applications are studied in the field of nuclear engineering."
text2 = "Nuclear physics is the branch of physics that studies atomic nuclei and their constituents and interactions. Examples of nuclear interactions or nuclear reactions include radioactive decay, nuclear fusion and fission. In this article, let us study nuclear physics, nuclear physics theory, nuclear force, and radioactivity in detail."

similarity_score = compute_similarity(text1, text2)
print(f"Similarity Score: {similarity_score:.2f}")
threshold = 0.5
if similarity_score > threshold:
    print("The texts are similar (potential plagiarism).")
else:
    print("The texts are not similar.")
