from flask import Flask, render_template,request,redirect,url_for,session
import cv2
import base64
from werkzeug.utils import secure_filename
import cartoonizer as cn
import os

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z^^%]/'
app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__))+ '/image'
app.config['TEMPLATE_FOLDER'] = os.path.realpath('.') + '/templates'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['image'] = filename
            return redirect(url_for('convert'))

    return render_template('upload.html')

@app.route('/convert')
def convert():
    if session:
        imgname = session['image']
        resname = 'temp'+session['image']
        imgdir = app.config['UPLOAD_FOLDER']+'/'+imgname
        resdir = app.config['UPLOAD_FOLDER']+'/'+resname
        img = cn.read(imgdir)
        k = cn.elbow(img)
        edges = cn.edgemask(img)
        img_kmeans = cn.cluster(img, k)
        c = cn.cartoon(img_kmeans, edges)
        cv2.imwrite(resdir, c)

        op = open(imgdir, 'rb')
        ip = base64.b64encode(op.read()).decode()
        op = open(resdir, 'rb')
        rp = base64.b64encode(op.read()).decode()

        session.clear()

        return render_template('convert.html',ip=ip, rp=rp)

    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(debug=True)