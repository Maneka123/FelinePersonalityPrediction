from flask import Flask, render_template, request, redirect, url_for, flash
import mysql.connector
import base64

app = Flask(__name__)
app.secret_key = '123'  # Change this to a random secret key

# Ensure you have the MySQL server running and the specified database and user exist.
mydb = mysql.connector.connect(
    host="localhost",
    port="3307",  # Default MySQL port
    user="root",
    password="",  # Use your MySQL root password
    database="animal_data"
)

mycursor = mydb.cursor(buffered=True)

mycursor.execute("CREATE TABLE IF NOT EXISTS cats (id INT AUTO_INCREMENT PRIMARY KEY, image LONGBLOB, breed VARCHAR(100), gender VARCHAR(10), age VARCHAR(10), stamina VARCHAR(50), emotion VARCHAR(50))")

# Admin password
ADMIN_PASSWORD = "123"

def is_admin(request):
    return request.form.get("admin_password") == ADMIN_PASSWORD

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if is_admin(request):
            if 'add_cat' in request.form:
                image_file = request.files['image']
                image_data = image_file.read()
                breed = request.form['breed']
                gender = request.form['gender']
                age = request.form['age']
                stamina = request.form['stamina']
                emotion = request.form['emotion']

                sql = "INSERT INTO cats (image, breed, gender, age, stamina, emotion) VALUES (%s, %s, %s, %s, %s, %s)"
                val = (image_data, breed, gender, age, stamina, emotion)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat added successfully!', 'success')
                return redirect(url_for('index'))

            elif 'edit_cat' in request.form:
                cat_id = request.form['cat_id']
                breed = request.form['edit_breed']
                gender = request.form['edit_gender']
                age = request.form['edit_age']
                stamina = request.form['edit_stamina']
                emotion = request.form['edit_emotion']

                sql = "UPDATE cats SET breed = %s, gender = %s, age = %s, stamina = %s, emotion = %s WHERE id = %s"
                val = (breed, gender, age, stamina, emotion, cat_id)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat updated successfully!', 'success')
                return redirect(url_for('index'))

            elif 'delete_cat' in request.form:
                cat_id = request.form['cat_id']
                sql = "DELETE FROM cats WHERE id = %s"
                val = (cat_id,)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat deleted successfully!', 'success')
                return redirect(url_for('index'))
        else:
            flash('Unauthorized access! Please enter admin password.', 'error')
            return redirect(url_for('index'))

    mycursor.execute("SELECT * FROM cats")
    cats = mycursor.fetchall()
    cat_records = []
    for cat in cats:
        cat_record = list(cat)
        cat_image_encoded = base64.b64encode(cat[1]).decode('utf-8')
        cat_record[1] = f"data:image/jpeg;base64,{cat_image_encoded}"
        cat_records.append(cat_record)
    return render_template('index2.html', cats=cat_records)

if __name__ == '__main__':
    app.run(debug=True)
