Classification of email if it is spam or not spam.

Steps Done During Preprocessing:

1)Converting into lower case.

2)Remove Punctuation.

3)Removing Stopwords.

4)Stemming

5)Count Vectorization

Model Building :

svc = SVC(kernel= 'sigmoid', gamma=1.0)

knc = KNeighborsClassifier()

mnb = MultinomialNB()

bnb = BernoulliNB()

gnb = GaussianNB()

dc = DecisionTreeClassifier(max_depth=5)

rc = LogisticRegression(solver='liblinear', penalty='l1')

rfc = RandomForestClassifier(n_estimators=50, random_state=2)

abc = AdaBoostClassifier(n_estimators=50, random_state=2)

bc = BaggingClassifier(n_estimators =50, random_state=2)

etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)

xgb = XGBClassifier(n_estimators=50, random_state=2)

