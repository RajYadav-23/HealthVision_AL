import sys
from app import create_app

app = create_app()
app.config['WTF_CSRF_ENABLED'] = False
app.config['TESTING'] = True

routes = [
    '/login', '/register',
    '/dashboard', '/explorer', '/visualization', '/preprocessing',
    '/ann', '/cnn', '/history', '/analytics', '/about', '/profile',
    '/ann/train/status', '/nonexistent'
]

all_ok = True
with app.test_client() as c:
    for r in routes:
        try:
            status = c.get(r).status_code
            ok = status in (200, 302, 404)
            print(f'  {"OK  " if ok else "FAIL"} {r} => {status}')
            if not ok:
                all_ok = False
        except Exception as e:
            print(f'  ERR  {r} => {e}')
            all_ok = False

print()
print('All routes OK' if all_ok else 'SOME ROUTES FAILED')
sys.exit(0 if all_ok else 1)
