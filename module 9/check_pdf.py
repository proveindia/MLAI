import PyPDF2
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Extract detailed topic coverage from QRG
with open('M09-QRG.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    full_text = ''
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    
    print('=== MODULE 9 COVERAGE CHECK ===\n')
    
    # Topics to verify
    checks = {
        'Polynomial regression': ['polynomial', 'degree'],
        'Cross validation': ['cross validation', 'cv='],
        'GridSearchCV': ['gridsearchcv', 'grid'],
        'Hyperparameter': ['hyperparameter', 'alpha'],
        'Scaling': ['standardscaler', 'scaling'],
        'Loss/MSE': ['loss', 'mse', 'mean squared'],
        'Ridge': ['ridge', 'l2'],
        'Regularization': ['regularization', 'penalty'],
        'Lasso': ['lasso', 'l1'],
        'Feature Selection': ['feature selection', 'sequential']
    }
    
    full_lower = full_text.lower()
    for topic, keywords in checks.items():
        count = sum(full_lower.count(kw.lower()) for kw in keywords)
        status = 'YES' if count > 3 else 'MINIMAL'
        print(f'{status:8} - {topic:20} ({count} mentions)')
