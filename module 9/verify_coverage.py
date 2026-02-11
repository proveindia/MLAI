import PyPDF2
import re

# Extract all section headers from SUMMARY.md
with open('SUMMARY.md', 'r', encoding='utf-8') as f:
    summary_text = f.read()

# Extract from QRG PDF
with open('M09-QRG.pdf', 'rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    qrg_text = ''
    for page in pdf_reader.pages:
        qrg_text += page.extract_text()

print('='*80)
print('MODULE 9 CONTENT VERIFICATION')
print('='*80)

# Define key topics from QRG that should be in summary
required_topics = {
    'Polynomial Features on Multidimensional Data': {
        'keywords': ['polynomial', 'degree', 'polynomialfeatures'],
        'found': False
    },
    'Sequential Feature Selection (SFS)': {
        'keywords': ['sequential', 'sfs', 'feature selection'],
        'found': False
    },
    'Regularization Concept': {
        'keywords': ['regularization', 'complexity', 'penalty'],
        'found': False
    },
    'Ridge Regression (L2)': {
        'keywords': ['ridge', 'l2', 'beta squared'],
        'found': False
    },
    'Lasso Regression (L1)': {
        'keywords': ['lasso', 'l1', 'absolute value'],
        'found': False
    },
    'Alpha Parameter': {
        'keywords': ['alpha', 'lambda', 'regularization strength'],
        'found': False
    },
    'StandardScaler / Feature Scaling': {
        'keywords': ['standardscaler', 'scaling', 'normalization'],
        'found': False
    },
    'GridSearchCV': {
        'keywords': ['gridsearchcv', 'grid search', 'hyperparameter tuning'],
        'found': False
    },
    'Cross Validation': {
        'keywords': ['cross validation', 'k-fold', 'cv'],
        'found': False
    },
    'MSE / Loss Function': {
        'keywords': ['mse', 'mean squared error', 'loss'],
        'found': False
    }
}

# Check summary for each topic
summary_lower = summary_text.lower()
for topic, info in required_topics.items():
    for keyword in info['keywords']:
        if keyword.lower() in summary_lower:
            info['found'] = True
            break

print('\n📊 COVERAGE ANALYSIS:\n')
for topic, info in required_topics.items():
    status = '✅ COVERED' if info['found'] else '❌ MISSING'
    print(f'{status:12} - {topic}')

# Count sections in summary
sections = re.findall(r'^## \d+\. (.+)$', summary_text, re.MULTILINE)
print(f'\n📚 Summary has {len(sections)} main sections:')
for i, section in enumerate(sections, 1):
    print(f'   {i}. {section}')

# Additional topics mentioned in QRG
print('\n🔍 ADDITIONAL QRG TOPICS TO CHECK:\n')
qrg_topics = [
    ('Overfitting vs Underfitting', ['overfit', 'underfit']),
    ('Dev Set / Train Set', ['dev set', 'development set']),
    ('Recursive Feature Elimination (RFE)', ['rfe',  'recursive']),
    ('Wrapper vs Filter vs Embedded Methods', ['wrapper', 'filter', 'embedded']),
]

for topic, keywords in qrg_topics:
    found_in_summary = any(kw.lower() in summary_lower for kw in keywords)
    found_in_qrg = any(kw.lower() in qrg_text.lower() for kw in keywords)
    
    if found_in_qrg:
        status = '✅' if found_in_summary else '⚠️'
        note = '' if found_in_summary else ' (in QRG but less detail in summary)'
        print(f'{status} {topic}{note}')

print('\n' + '='*80)
print('VERDICT:')
print('='*80)

covered_count = sum(1 for t in required_topics.values() if t['found'])
total_count = len(required_topics)
coverage_pct = (covered_count / total_count) * 100

print(f'\nCoverage: {covered_count}/{total_count} topics ({coverage_pct:.0f}%)')

if coverage_pct >= 90:
    print('✅ EXCELLENT - Summary comprehensively covers all major topics!')
elif coverage_pct >= 70:
    print('✓ GOOD - Summary covers most topics, minor gaps')
else:
    print('⚠️ NEEDS IMPROVEMENT - Some key topics missing')
