import hashlib
from flask import Flask, jsonify, request

app = Flask(__name__)

users = []

class User:
    def __init__(self, username, email, password, profile_picture=None, cultural_background="", interests=""):
        self.username = username
        self.email = email
        self.password = password  # hashed password
        self.profile_picture = profile_picture
        self.cultural_background = cultural_background
        self.interests = interests

def hash_password(password):
    """Hash a password using MD5 (for simplicity in this example)"""
    return hashlib.md5(password.encode()).hexdigest()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    cultural_background = data.get('cultural_background', '')
    interests = data.get('interests', '')
    profile_picture = data.get('profile_picture', '')
    
    # Validate required fields
    if not username or not email or not password:
        return jsonify({'error': 'Username, email, and password are required'}), 400
    
    # Check for existing username or email
    if any(user.username == username for user in users) or any(user.email == email for user in users):
        return jsonify({'error': 'Username or email already exists'}), 400
    
    # Hash the password
    hashed_password = hash_password(password)
    
    new_user = User(username, email, hashed_password, profile_picture, cultural_background, interests)
    users.append(new_user)
    return jsonify({
        'message': 'User registered successfully',
        'user': {
            'username': new_user.username,
            'email': new_user.email,
            'profile_picture': new_user.profile_picture,
            'cultural_background': new_user.cultural_background,
            'interests': new_user.interests
        }
    }), 201

@app.route('/profile/update', methods=['PUT'])
def update_profile():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    user = next((u for u in users if u.username == username), None)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Update fields
    user.cultural_background = data.get('cultural_background', user.cultural_background)
    user.interests = data.get('interests', user.interests)
    user.profile_picture = data.get('profile_picture', user.profile_picture)
    
    return jsonify({
        'message': 'Profile updated successfully',
        'user': {
            'username': user.username,
            'email': user.email,
            'profile_picture': user.profile_picture,
            'cultural_background': user.cultural_background,
            'interests': user.interests
        }
    }), 200

if __name__ == '__main__':
    app.run(debug=True)