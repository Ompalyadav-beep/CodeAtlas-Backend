
import os
import re
import requests
import base64
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from markdown import markdown
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy

# --- Step 1: Load API Keys & Configure ---
load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ideas.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# --- Database Models ---
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    repo_name = db.Column(db.String(100), nullable=False)
    idea = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    comments = db.relationship('Comment', backref='post', lazy=True, cascade="all, delete-orphan")

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)

# Securely load API keys
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GITHUB_TOKEN or not GEMINI_API_KEY:
    raise ValueError("🔴 Critical Error: GITHUB_TOKEN or GEMINI_API_KEY not found in .env file.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
except Exception as e:
    raise RuntimeError(f"🔴 Error configuring Gemini AI: {e}")

# --- Helper Functions ---
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2).strip()
    return None, None

def get_github_readme(owner, repo_name):
    url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        return content, None
    except requests.exceptions.HTTPError as err:
        return None, f"Could not fetch README. (HTTP Error: {err.response.status_code})"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def summarize_readme_with_gemini(readme_content):
    if not readme_content:
        return None, "README content is empty."
    prompt = f"As a senior software engineer, analyze the following README and provide a concise summary in Markdown format, including project purpose, key features, and technology stack.\n---{readme_content}"
    try:
        response = model.generate_content(prompt)
        return markdown(response.text), None
    except Exception as e:
        return None, f"Error generating summary: {e}"

def get_github_file_structure(owner, repo_name):
    repo_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        repo_info = requests.get(repo_url, headers=headers, timeout=10)
        repo_info.raise_for_status()
        default_branch = repo_info.json()['default_branch']
        tree_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}?recursive=1"
        tree_response = requests.get(tree_url, headers=headers, timeout=15)
        tree_response.raise_for_status()
        files = [item['path'] for item in tree_response.json()['tree'] if item['type'] == 'blob']
        return "\n".join(files[:200]), None
    except Exception as e:
        return None, f"Could not fetch file structure: {e}"

def analyze_structure_with_gemini(file_structure):
    if not file_structure:
        return None, "File structure is empty."
    prompt = f"You are a principal software architect. Based on the file structure, provide a high-level analysis in Markdown format, including likely architecture, key components, and code organization.\n---{file_structure}"
    try:
        response = model.generate_content(prompt)
        return markdown(response.text), None
    except Exception as e:
        return None, f"Error generating analysis: {e}"

def get_setup_guide_with_gemini(readme, file_structure):
    prompt = f"Based on the README and file structure, provide a step-by-step setup guide in Markdown format.\n---README---\n{readme}\n---FILE STRUCTURE---\n{file_structure}"
    try:
        response = model.generate_content(prompt)
        return markdown(response.text), None
    except Exception as e:
        return None, f"Error generating setup guide: {e}"


# --- API Routes ---
@app.route('/api/analyze', methods=['POST'])
def analyze_repo_route():
    data = request.get_json()
    repo_url = data.get('repo_url')
    if not repo_url:
        return jsonify({"error": "repo_url is required"}), 400

    owner, repo_name = parse_github_url(repo_url)
    if not owner or not repo_name:
        return jsonify({"error": "Invalid GitHub URL"}), 400

    readme_content, error = get_github_readme(owner, repo_name)
    if error:
        return jsonify({"error": error}), 500
    
    readme_summary, error = summarize_readme_with_gemini(readme_content)
    if error:
        return jsonify({"error": error}), 500

    file_structure, error = get_github_file_structure(owner, repo_name)
    if error:
        return jsonify({"error": error}), 500

    structure_analysis, error = analyze_structure_with_gemini(file_structure)
    if error:
        return jsonify({"error": error}), 500
        
    setup_guide, error = get_setup_guide_with_gemini(readme_content, file_structure)
    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "readme_summary": readme_summary,
        "structure_analysis": structure_analysis,
        "setup_guide": setup_guide,
    })

@app.route('/api/trending', methods=['GET'])
def trending_repos_route():
    search_query = request.args.get('search_query', default=None, type=str)
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    q = [f"{search_query}"] if search_query else []
    q.append(f"created:>{(datetime.utcnow() - timedelta(days=730)).strftime('%Y-%m-%d')}")
    query = '+'.join(q)
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=12"
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        items = resp.json().get('items', [])
        result = [{
            'name': r['full_name'],
            'url': r['html_url'],
            'stars': r['stargazers_count'],
            'description': r['description'] or '',
            'forks': r['forks_count'],
        } for r in items]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error searching repos: {e}"}), 500

@app.route('/api/posts', methods=['GET'])
def get_posts():
    try:
        posts = Post.query.order_by(Post.timestamp.desc()).all()
        return jsonify([{
            'id': post.id,
            'repo_name': post.repo_name,
            'idea': post.idea,
            'timestamp': post.timestamp.strftime('%Y-%m-%d %H:%M'),
            'comments_count': len(post.comments)
        } for post in posts])
    except Exception as e:
        return jsonify({"error": "Failed to fetch posts"}), 500

@app.route('/api/posts', methods=['POST'])
def add_post():
    data = request.get_json()
    repo_name = data.get('repo_name')
    idea = data.get('idea')
    if repo_name and idea:
        try:
            new_post = Post(repo_name=repo_name, idea=idea)
            db.session.add(new_post)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Idea posted successfully!'}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': 'Database error.'}), 500
    return jsonify({'success': False, 'message': 'Missing repository name or idea.'}), 400

# --- Run the App ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 