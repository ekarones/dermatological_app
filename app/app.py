from flask import Flask, request, render_template, redirect, url_for, abort
import sqlite3
import os
import base64
import joblib
import numpy as np
import cv2

base_dir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(base_dir, "database", "my_database.db")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model_filename = "svm_model.joblib"
svm_model = joblib.load(model_filename)

disease_folders = ["PSORIASIS", "ROSACEA", "SARPULLIDO", "VITILIGIO"]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login_patient", methods=["POST", "GET"])
def login_patient():
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        if request.method == "POST":
            documento = request.form["documento"]
            contrasena = request.form["contrasena"]

            cursor.execute(
                "SELECT * FROM Patients WHERE document=? AND password=?",
                (documento, contrasena),
            )
            paciente = cursor.fetchone()

            if paciente:
                return redirect(url_for("interface_patient", paciente_id=paciente[0]))

    except Exception as e:
        print("Error al conectar a la base de datos:", e)

    finally:
        if conn:
            conn.close()

    return render_template("login_patient.html")


@app.route("/interface_patient/<int:paciente_id>", methods=["POST", "GET"])
def interface_patient(paciente_id):
    return render_template("interface_patient.html", paciente_id=paciente_id)


@app.route("/login_staff", methods=["POST", "GET"])
def login_staff():
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        if request.method == "POST":
            documento = request.form["documento"]
            contrasena = request.form["contrasena"]

            cursor.execute(
                "SELECT * FROM Doctors WHERE document=? AND password=?",
                (documento, contrasena),
            )
            doctor = cursor.fetchone()

            if doctor:
                return redirect(url_for("interface_staff"))

    except Exception as e:
        print("Error al conectar a la base de datos:", e)

    finally:
        if conn:
            conn.close()

    return render_template("login_staff.html")


@app.route("/interface_staff", methods=["POST", "GET"])
def interface_staff():
    return render_template("interface_staff.html")


@app.route("/add_patient", methods=["POST", "GET"])
def add_patient():
    if request.method == "POST":
        nombre = request.form["nombre"]
        apellido = request.form["apellido"]
        documento = request.form["documento"]
        telefono = request.form["telefono"]
        date = request.form["fecha_nacimiento"]
        correo = request.form["correo"]
        contrasena = request.form["contrasena"]
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Patients (name, lastName,document,telf,date,email,password) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (nombre, apellido, documento, telefono, date, correo, contrasena),
        )
        conn.commit()
        conn.close()
        return redirect("/show_patients")
    return render_template("add_patient.html")


@app.route("/add_doctor", methods=["POST", "GET"])
def add_doctor():
    if request.method == "POST":
        nombre = request.form["nombre"]
        apellido = request.form["apellido"]
        descripcion = request.form["descripcion"]
        dni = request.form["documento"]
        contrasena = request.form["contrasena"]
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Doctors (name, lastName, description, document, password) VALUES ( ?, ?, ?, ?, ?)",
            (nombre, apellido, descripcion, dni, contrasena),
        )
        conn.commit()
        conn.close()
        return redirect("/show_doctors")
    return render_template("add_doctor.html")


@app.route("/show_patients")
def show_patients():
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Patients")
    resultados = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("show_patients.html", pacientes=resultados)


@app.route("/show_doctors")
def show_doctors():
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Doctors")
    resultados = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("show_doctors.html", doctores=resultados)


@app.route("/predict_section/<int:paciente_id>", methods=["POST", "GET"])
def predict_section(paciente_id):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, lastName FROM Patients WHERE id = ?", (paciente_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        paciente_name = result[0]
        paciente_lastname = result[1]
        nombre = paciente_name + " " + paciente_lastname

        return render_template(
            "predict_section.html", paciente_name=nombre, paciente_id=paciente_id
        )
    else:
        abort(404)


@app.route("/mostrar_imagen/<int:paciente_id>", methods=["POST", "GET"])
def upload_image(paciente_id):
    if "file" not in request.files:
        return render_template("index.html")

    if "file" not in request.files:
        return render_template("index.html")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html")

    if file and allowed_file(file.filename):
        uploaded_image = file.read()
        encoded_image = base64.b64encode(uploaded_image).decode("utf-8")

        image = cv2.imdecode(
            np.fromstring(uploaded_image, np.uint8), cv2.IMREAD_UNCHANGED
        )
        resized_image = cv2.resize(image, (256, 256))
        flattened_image = resized_image.reshape(-1)
        predicted_label = svm_model.predict([flattened_image])
        predicted_class = disease_folders[int(predicted_label)]
        if predicted_class == "PSORIASIS":
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Patients SET disease = ? WHERE id = ?",
                (predicted_class, paciente_id),
            )
            conn.commit()
            conn.close()
            return render_template(
                "case_psoriasis.html",
                mensaje=predicted_class,
                image=encoded_image,
                paciente_id=paciente_id,
            )
        if predicted_class == "ROSACEA":
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Patients SET disease = ? WHERE id = ?",
                (predicted_class, paciente_id),
            )
            conn.commit()
            conn.close()
            return render_template(
                "case_rosacea.html",
                mensaje=predicted_class,
                image=encoded_image,
                paciente_id=paciente_id,
            )
        if predicted_class == "SARPULLIDO":
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Patients SET disease = ? WHERE id = ?",
                (predicted_class, paciente_id),
            )
            conn.commit()
            conn.close()
            return render_template(
                "case_sarpullido.html",
                mensaje=predicted_class,
                image=encoded_image,
                paciente_id=paciente_id,
            )
        if predicted_class == "VITILIGIO":
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Patients SET disease = ? WHERE id = ?",
                (predicted_class, paciente_id),
            )
            conn.commit()
            conn.close()
            return render_template(
                "case_vitiligio.html",
                mensaje=predicted_class,
                image=encoded_image,
                paciente_id=paciente_id,
            )
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
