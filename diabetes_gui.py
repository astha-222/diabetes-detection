import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox, font

# Load the trained model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Predict function
def predict_diabetes():
    try:
        input_data = [
            float(preg_var.get()),
            float(glucose_var.get()),
            float(bp_var.get()),
            float(skin_var.get()),
            float(insulin_var.get()),
            float(bmi_var.get()),
            float(dpf_var.get()),
            float(age_var.get())
        ]

        import pandas as pd

        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        input_df = pd.DataFrame([input_data], columns=columns)
        prediction = loaded_model.predict(input_df)

        if prediction[0] == 0:
            result_label.config(text="The person is NOT Diabetic", fg="green")
        else:
            result_label.config(text="The person IS Diabetic", fg="red")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction System")
root.geometry("500x600")
root.configure(bg="#f0f4f8")

# Font styles
title_font = font.Font(family='Helvetica', size=16, weight='bold')
label_font = font.Font(family='Helvetica', size=10)

# Title
tk.Label(root, text="Diabetes Prediction System", font=title_font, bg="#f0f4f8", fg="#333").pack(pady=20)

# Input Fields
fields = [
    ("Pregnancies", "preg_var"),
    ("Glucose", "glucose_var"),
    ("Blood Pressure", "bp_var"),
    ("Skin Thickness", "skin_var"),
    ("Insulin", "insulin_var"),
    ("BMI", "bmi_var"),
    ("Diabetes Pedigree Function", "dpf_var"),
    ("Age", "age_var"),
]

entries = {}
for idx, (label_text, var_name) in enumerate(fields):
    tk.Label(root, text=label_text, font=label_font, bg="#f0f4f8").pack()
    var = tk.StringVar()
    entries[var_name] = var
    tk.Entry(root, textvariable=var, width=30, bd=2, relief="groove").pack(pady=5)

# Bind variables
preg_var = entries["preg_var"]
glucose_var = entries["glucose_var"]
bp_var = entries["bp_var"]
skin_var = entries["skin_var"]
insulin_var = entries["insulin_var"]
bmi_var = entries["bmi_var"]
dpf_var = entries["dpf_var"]
age_var = entries["age_var"]

# Predict Button
tk.Button(root, text="Predict", command=predict_diabetes, bg="#007acc", fg="white", padx=20, pady=10, font=label_font).pack(pady=20)

# Result Label
result_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"), bg="#f0f4f8")
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()

