import matplotlib.pyplot as plt
import pandas as pd
import numpy
from django.http import HttpResponse
from django.shortcuts import render
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from django.contrib import messages
import os

from app.forms import Reg
from .models import Register

def convert(string):
    li =list(string.split("\n"))
    return li
def home(request):
    return render(request,'home.html')
def upload(request):
    return render(request, 'upload.html', {"Reg":Reg()})
def Loading(request):
    images=request.POST.get("file")
    images=request.FILES["images"]
    Register(images=images).save()
    qr=Register.objects.filter()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
    media_file = MEDIA_ROOT + "\\files\\" + str(images)
    try:
        file_name, file_extension = os.path.splitext(media_file)
        if file_extension == '.csv':
            response: HttpResponse = render(request, "Loading.html", {'file': images})
            response.set_cookie('file', images)
            return response
        else:
            os.remove(media_file)
            messages.warning(request, 'The uploaded file is not a CSV file')
            return render(request, 'upload.html', {"Reg": Reg()})
    except:
        messages.warning(request, 'The file name must be single word with out spaces')
        return render(request, 'upload.html', {"Reg": Reg()})
def result(request):
    try:
        a = request.COOKIES['file']
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
        media_file = MEDIA_ROOT + "\\files\\"+a
        STATIC_DIR = os.path.join(BASE_DIR, 'app\\static')
        staticfile = STATIC_DIR + '\\img\\pie.png'
        df = pd.read_csv(media_file)
        X = df.iloc[:, 0:8]
        y = df.iloc[:, 8]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=195)
        xi = X_train.shape[0]
        xj = X_test.shape[0]
        slices = [xi, xj]
        a = 'Train ' + str(xi) + ' data'
        b = 'Test ' + str(xj) + ' data'
        labels = [a, b]
        sizes = [a, b]
        fig1, ax1 = plt.subplots()
        ax1.pie(slices, autopct='%1.1f%%', explode=(0, 0.1), labels=labels, shadow=True, startangle=90)
        plt.title('pie chart')
        plt.legend(title="train data", loc="lower right")
        ax1.axis('equal')
        plt.tight_layout()
        plt.savefig(staticfile)
        mlp = MLPClassifier(random_state=200)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        z = format(mlp.score(X_train, y_train), '.2f')
        p = format(mlp.score(X_test, y_test), '.2f')
        q = metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
        co = convert(q)
        r = metrics.confusion_matrix(y_test, y_pred)
        dtree_model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
        y_pred = dtree_model.predict(X_test)
        ab = format(dtree_model.score(X_train, y_train), '.2f')
        bc = format(dtree_model.score(X_test, y_test), '.2f')
        cd = metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
        df = metrics.confusion_matrix(y_test, y_pred)
        knn = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        kz = format(knn.score(X_train, y_train), '.2f')
        kp = format(knn.score(X_test, y_test), '.2f')
        kq = metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
        kr = metrics.confusion_matrix(y_test, y_pred)
        gnb = GaussianNB().fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        nz = format(gnb.score(X_train, y_train), '.2f')
        np = format(gnb.score(X_test, y_test), '.2f')
        nq = metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
        nr = metrics.confusion_matrix(y_test, y_pred)
        svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
        y_pred = svm_model_linear.predict(X_test)
        sz = format(svm_model_linear.score(X_train, y_train), '.2f')
        sp = format(svm_model_linear.score(X_test, y_test), '.2f')
        sq = metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
        sr = metrics.confusion_matrix(y_test, y_pred)
        lg = LogisticRegression(random_state=0)
        lg.fit(X_train, y_train)
        y_pred = lg.predict(X_test)
        lz = format(lg.score(X_train, y_train), '.2f')
        lp = format(lg.score(X_test, y_test), '.2f')
        lq = metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
        lr = metrics.confusion_matrix(y_test, y_pred)
        maxi = max(p, bc, kp, np, sp, lp)
        if maxi == p:
            alog = "Nural Networks"
        elif maxi == bc:
            alog = "Decision Tree"
        elif maxi == kp:
            alog = "K-nearest neighbors"
        elif maxi == np:
            alog = "Naive Bayes"
        elif maxi == sp:
            alog = "Support Vector Machines"
        else:
            alog = "LogisticRegression"
        mlp = {'train': str(z), 'test': str(p), 'matrix': str(r), 'report': q, 'dtrain': str(ab), 'dtest': str(bc),
               'dmatrix': str(r), 'dreport': cd, 'co':co,
               'ktrain': str(kz), 'ktest': str(kp), 'kmatrix': str(kr), 'kreport': kq, 'ntrain': str(nz),
               'ntest': str(np), 'nmatrix': str(nr), 'nreport': nq,
               'strain': str(sz), 'stest': str(sp), 'smatrix': str(sr), 'sreport': sq, 'ltrain': str(lz),
               'ltest': str(lp), 'lmatrix': str(lr), 'lreport': lq,
               'maxi': maxi, 'alog': alog}
        return render(request, 'result.html', {'mlp': mlp})
    except ValueError:
        try:
            a = request.COOKIES['file']
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
            media_file = MEDIA_ROOT + "\\files\\"+a
            STATIC_DIR = os.path.join(BASE_DIR, 'app\\static')
            staticfile = STATIC_DIR + '\\img\\pie.png'
            df = pd.read_csv(media_file)
            k = list(df.columns)
            df = df.set_index(k[0])
            l1 = list(df.columns == 'Class')
            if True in l1:
                pass
            else:
                raise NameError('hi')
            df.replace('?', numpy.nan, inplace=True)
            df.dropna(inplace=True)
            class_counts = df.groupby('Class').size()
            X = df.drop('Class', axis=1)
            y = df['Class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
            a = X_train.shape[0]
            b = X_train.shape[1]
            c = X_test.shape[0]
            d = X_test.shape[1]
            slices = [a, c]
            a = 'Train ' + str(a) + ' data'
            b = 'Test ' + str(b) + ' data'
            labels = [a, b]
            sizes = [a, b]
            fig1, ax1 = plt.subplots()
            ax1.pie(slices, autopct='%1.1f%%', explode=(0, 0.1), labels=labels, shadow=True, startangle=90)
            plt.title('pie chart')
            plt.legend(title="train data", loc="lower right")
            ax1.axis('equal')
            plt.tight_layout()
            plt.savefig(staticfile)
            dt = DecisionTreeClassifier(random_state=100, max_depth=3, min_samples_leaf=5)
            dt.fit(X_train, y_train)
            pred = dt.predict(X_test)
            dt.score(X_train, y_train)
            e = accuracy_score(y_test, pred) * 100
            f = format(dt.score(X_train, y_train), '.2f')
            g = format(dt.score(X_test, y_test), '.2f')
            count_misclassified = (y_test != pred).sum()
            h = confusion_matrix(y_test, pred)
            dtc = classification_report(y_test, pred)
            pd.DataFrame(confusion_matrix(y_test, pred),
                         columns=['Benign', 'Malignant'],
                         index=['True Benign', 'True Malignan']
                         )
            mlp = MLPClassifier(random_state=200)
            mlp.fit(X_train, y_train)
            pred = mlp.predict(X_test)
            mlp.score(X_train, y_train)
            j = accuracy_score(y_test, pred) * 100
            k = format(mlp.score(X_train, y_train), '.2f')
            l = format(mlp.score(X_test, y_test), '.2f')
            mcount_misclassified = (y_test != pred).sum()
            m = confusion_matrix(y_test, pred)
            n = classification_report(y_test, pred)
            pd.DataFrame(confusion_matrix(y_test, pred),
                         columns=['Benign', 'Malignant'],
                         index=['True Benign', 'True Malignan']
                         )
            knn = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
            pred = knn.predict(X_test)
            knn.score(X_train, y_train)
            o = accuracy_score(y_test, pred) * 100
            p = format(knn.score(X_train, y_train), '.2f')
            q = format(knn.score(X_test, y_test), '.2f')
            kcount_misclassified = (y_test != pred).sum()
            r = confusion_matrix(y_test, pred)
            s = classification_report(y_test, pred)
            pd.DataFrame(confusion_matrix(y_test, pred),
                         columns=['Benign', 'Malignant'],
                         index=['True Benign', 'True Malignan']
                         )
            gnb = GaussianNB().fit(X_train, y_train)
            pred = gnb.predict(X_test)
            gnb.score(X_train, y_train)
            t = accuracy_score(y_test, pred) * 100
            x = format(gnb.score(X_train, y_train), '.2f')
            y = format(gnb.score(X_test, y_test), '.2f')
            gcount_misclassified = (y_test != pred).sum()
            z = confusion_matrix(y_test, pred)
            gn = classification_report(y_test, pred)
            pd.DataFrame(confusion_matrix(y_test, pred),
                         columns=['Benign', 'Malignant'],
                         index=['True Benign', 'True Malignan']
                         )
            svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
            pred = svm_model_linear.predict(X_test)
            svm_model_linear.score(X_train, y_train)
            ab = accuracy_score(y_test, pred) * 100
            cd = format(svm_model_linear.score(X_train, y_train), '.2f')
            ef = format(svm_model_linear.score(X_test, y_test), '.2f')
            svmcount_misclassified = (y_test != pred).sum()
            gh = confusion_matrix(y_test, pred)
            ij = classification_report(y_test, pred)
            pd.DataFrame(confusion_matrix(y_test, pred),
                         columns=['Benign', 'Malignant'],
                         index=['True Benign', 'True Malignan']
                         )
            lg = LogisticRegression(random_state=0)
            lg.fit(X_train, y_train)
            pred = lg.predict(X_test)
            lg.score(X_train, y_train)
            kl = accuracy_score(y_test, pred) * 100
            mn = format(lg.score(X_train, y_train), '.2f')
            op = format(lg.score(X_test, y_test), '.2f')
            lgcount_misclassified = (y_test != pred).sum()
            qr = confusion_matrix(y_test, pred)
            st = classification_report(y_test, pred)
            pd.DataFrame(confusion_matrix(y_test, pred),
                         columns=['Benign', 'Malignant'],
                         index=['True Benign', 'True Malignan']
                         )
            maxi = max(e, j, o, t, ab, kl)
            if maxi == j:
                alog = "Nural Networks"
            elif maxi == e:
                alog = "Decision Tree"
            elif maxi == o:
                alog = "K-Nearest Neighbors"
            elif maxi == t:
                alog = "Naive Bayes"
            elif maxi == ab:
                alog = "Support Vector Machines"
            else:
                alog = "LogisticRegression"
            mlp = {'train': str(k), 'test': str(l), 'matrix': str(m), 'report': n, 'dtrain': str(f), 'dtest': str(g),
                   'dmatrix': str(h), 'dreport': dtc,
                   'ktrain': str(p), 'ktest': str(q), 'kmatrix': str(r), 'kreport': s, 'ntrain': str(x),
                   'ntest': str(y), 'nmatrix': str(z), 'nreport': gn,
                   'strain': str(cd), 'stest': str(ef), 'smatrix': str(gh), 'sreport': ij, 'ltrain': str(mn),
                   'ltest': str(op), 'lmatrix': str(qr), 'lreport': st,
                   'maxi': maxi, 'alog': alog}
            return render(request, 'result.html', {'mlp': mlp})
        except NameError:
            messages.warning(request, 'Please change the class label name as Class')
            return render(request, 'upload.html', {"Reg": Reg()})
    except:
        messages.warning(request,'Please check the file it raising some error')
        return render(request,'upload.html',{"Reg":Reg()})
