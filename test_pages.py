import sys
from app import create_app
from database.models import db, User

app = create_app()
app.config['WTF_CSRF_ENABLED'] = False
app.config['TESTING'] = True

protected = [
    '/dashboard', '/explorer', '/visualization', '/preprocessing',
    '/ann', '/cnn', '/history', '/analytics', '/about', '/profile',
    '/ann/train/status'
]

all_ok = True
with app.test_client() as c:
    # Create test user
    with app.app_context():
        if not User.query.filter_by(email='test@hv.com').first():
            u = User(username='testuser', email='test@hv.com')
            u.set_password('test1234')
            db.session.add(u)
            db.session.commit()

    # Login
    r = c.post('/login', data={'email': 'test@hv.com', 'password': 'test1234'}, follow_redirects=True)
    print(f'  Login => {r.status_code}')

    # Test all protected pages
    for route in protected:
        try:
            resp = c.get(route, follow_redirects=True)
            ok = resp.status_code == 200
            print(f'  {"OK  " if ok else "FAIL"} {route} => {resp.status_code}')
            if not ok:
                all_ok = False
        except Exception as e:
            print(f'  ERR  {route} => {e}')
            all_ok = False

print()
print('All pages render OK' if all_ok else 'SOME PAGES FAILED')
sys.exit(0 if all_ok else 1)
