import sys, json
from app import create_app
from database.models import db, User

app = create_app()
app.config['WTF_CSRF_ENABLED'] = False
app.config['TESTING'] = True

all_ok = True

with app.test_client() as c:
    # Ensure test user exists
    with app.app_context():
        if not User.query.filter_by(email='test@hv.com').first():
            u = User(username='testuser', email='test@hv.com')
            u.set_password('test1234')
            db.session.add(u)
            db.session.commit()

    c.post('/login', data={'email': 'test@hv.com', 'password': 'test1234'}, follow_redirects=True)

    # Test ANN prediction
    pred_data = {
        'Pregnancies': '2', 'Glucose': '130', 'BloodPressure': '72',
        'SkinThickness': '25', 'Insulin': '100', 'BMI': '28.5',
        'DiabetesPedigreeFunction': '0.45', 'Age': '35'
    }
    r = c.post('/ann/predict', data=pred_data, follow_redirects=True)
    ok = r.status_code == 200 and b'Positive' in r.data or b'Negative' in r.data
    print(f'  {"OK  " if ok else "FAIL"} ANN predict => {r.status_code}')
    if not ok: all_ok = False

    # Test ANN page with charts
    r = c.get('/ann')
    ok = r.status_code == 200 and b'Accuracy' in r.data
    print(f'  {"OK  " if ok else "FAIL"} ANN charts loaded => {r.status_code}')
    if not ok: all_ok = False

    # Test dashboard shows real data
    r = c.get('/dashboard')
    ok = r.status_code == 200 and b'Total Predictions' in r.data
    print(f'  {"OK  " if ok else "FAIL"} Dashboard => {r.status_code}')
    if not ok: all_ok = False

    # Test history shows prediction
    r = c.get('/history')
    ok = r.status_code == 200
    print(f'  {"OK  " if ok else "FAIL"} History => {r.status_code}')
    if not ok: all_ok = False

    # Test analytics
    r = c.get('/analytics')
    ok = r.status_code == 200
    print(f'  {"OK  " if ok else "FAIL"} Analytics => {r.status_code}')
    if not ok: all_ok = False

    # Test training status endpoint
    r = c.get('/ann/train/status')
    ok = r.status_code == 200
    try:
        data = json.loads(r.data)
        ok = 'running' in data and 'progress' in data
    except:
        ok = False
    print(f'  {"OK  " if ok else "FAIL"} Train status JSON => {r.status_code}')
    if not ok: all_ok = False

    # Test 404
    r = c.get('/nonexistent')
    ok = r.status_code == 404 and b'404' in r.data
    print(f'  {"OK  " if ok else "FAIL"} 404 page => {r.status_code}')
    if not ok: all_ok = False

    # Test profile
    r = c.get('/profile')
    ok = r.status_code == 200 and b'testuser' in r.data
    print(f'  {"OK  " if ok else "FAIL"} Profile => {r.status_code}')
    if not ok: all_ok = False

print()
print('ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED')
sys.exit(0 if all_ok else 1)
