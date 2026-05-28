import math

SKILL_ALIASES = {
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android",
    "firebase": "firebase",
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma"
}

def get_skills(raw):
    tokens = [t.strip().lower() for t in raw.split(",")]
    result = []
    seen = set()
    for token in tokens:
        if token in SKILL_ALIASES:
            canonical = SKILL_ALIASES[token]
            if canonical not in seen:
                result.append(canonical)
                seen.add(canonical)
    return result
resumes = [
    ("01", "Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("02", "Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("03", "Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("04", "Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("05", "Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("06", "Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("07", "Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("08", "Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("09", "Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("10", "Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

for rid, name, raw in resumes:
    skills = get_skills(raw)
    print(rid, name, skills)

all_skills = set()
for rid, name, raw in resumes:
    skills = get_skills(raw)
    all_skills.update(skills)

vocab = sorted(all_skills)
print(vocab)
print("total skills:", len(vocab))

doc_freq = {}
for skill in vocab:
    count = 0
    for rid, name, raw in resumes:
        if skill in get_skills(raw):
            count += 1
    doc_freq[skill] = count

for skill in vocab:
    print(skill, "->", doc_freq[skill])


idf = {}
for skill in vocab:
    idf[skill] = math.log(10 / doc_freq[skill])

for skill in vocab:
    print(skill, "->", round(idf[skill], 4))

word_to_idx = {w: i for i, w in enumerate(vocab)}

def make_tfidf_vec(skills):
    N = len(skills)
    vec = [0.0] * len(vocab)
    for s in skills:
        tf = 1 / N
        vec[word_to_idx[s]] = tf * idf[s]
    return vec

resume_vecs = []
for rid, name, raw in resumes:
    skills = get_skills(raw)
    vec = make_tfidf_vec(skills)
    resume_vecs.append((name, vec))

print("vectors built for", len(resume_vecs), "resumes")

jds = [
    ("JD-1", "Kakao (ML Engineer)",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2", "Naver (Backend Engineer)",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis"),
    ("JD-3", "Line (Frontend Engineer)",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"),
]

def make_jd_vec(raw):
    skills = get_skills(raw)
    vec = [0.0] * len(vocab)
    for s in skills:
        if s in word_to_idx:
            vec[word_to_idx[s]] = 1.0
    return vec


jd1_vec = make_jd_vec(jds[0][2])
print("JD1 skills found in vocab:")
for i, val in enumerate(jd1_vec):
    if val > 0:
        print(vocab[i])

def cosine_sim(a, b):
    dot = sum(a[i] * b[i] for i in range(len(a)))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

for jd_id, jd_label, jd_raw in jds:
    jd_vec = make_jd_vec(jd_raw)
    scores = []
    for name, vec in resume_vecs:
        sim = cosine_sim(vec, jd_vec)
        scores.append((name, sim))
    scores.sort(key=lambda x: (-x[1], x[0]))
    print(jd_id, "—", jd_label)
    print(", ".join(f"{n}({s:.2f})" for n, s in scores[:3]))
    print()











