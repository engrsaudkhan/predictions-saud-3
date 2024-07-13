import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Graphical User Interface (GUI) for prediction of compressive strength of coral sand aggregate")
        master.configure(background="#f0f0f0")
        window_width = 750
        window_height = 740
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_cord = 0  # Start from the left edge of the screen
        y_cord = 0  # Start from the top edge of the screen
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        x_cord = 0
        y_cord = 0
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        main_heading = tk.Label(master, text="Graphical User Interface (GUI) for: \n Prediction of compressive strength of coral sand aggregate",
                               bg="#9b59b6", fg="#FFFFFF", font=("Helvetica", 16, "bold"), pady=10)
        main_heading.pack(side=tk.TOP, fill=tk.X)
        self.content_frame = tk.Frame(master, bg="#E8E8E8")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=50, anchor=tk.CENTER)
        self.canvas = tk.Canvas(self.content_frame, bg="#E8E8E8")
        self.scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#E8E8E8")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.TOP, fill="both", padx=10, pady=10, expand=True)
        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#FFFFFF", fg="#4A90E2", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.output_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both", padx=20, pady=20)
        heading = tk.Label(self.output_frame, text="Predictions", bg="#FFFFFF", fg="#4A90E2", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=2, pady=10)
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.input_frame.grid_columnconfigure(2, weight=1)
        self.create_entry("CSA Content:", 60, 7)
        self.create_entry("Immersion Period:", 2, 9)
        self.G2C0 = 5.15277535325175
        self.G3C5 = 9.27638468704486
        self.G3C3 = 4.53739360093925
        self.G4C1 = 6.0622265224757
        self.G4C5 = 10.0006711697009
        self.G1C6 = 11.0080760332607
        self.G2C2 = 5.96653065584277
        self.create_entry("Pressure:", 90, 1)
        self.create_entry("Particle Size:", 7.5, 3)
        self.create_entry("Particle Shape:", 4, 5)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.calculate_button_a = tk.Button(self.output_frame, text="Gene Expression Programming (GEP)", command=self.calculate_y_a,
                                          bg="red", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_a.grid(row=1, column=0, pady=10, padx=10)
        self.a_output_text_a = tk.Text(self.output_frame, height=2, width=30)
        self.a_output_text_a.grid(row=1, column=1, padx=10, pady=10)
        self.b_button_b = tk.Button(self.output_frame, text="Extreme Gradient Boosting (XGB)", command=self.calculate_b_b,
                                        bg="red", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.b_button_b.grid(row=2, column=0, pady=10, padx=10)
        self.b_output_text_b = tk.Text(self.output_frame, height=2, width=30)
        self.b_output_text_b.grid(row=2, column=1, padx=10, pady=10)
        developer_info = tk.Label(text="This GUI is developed by combined efforts of:\nMuhammad Saud Khan (khans28@myumanitoba.ca), University of Manitoba, Canada\nZohaib Mehmood (zoohaibmehmood@gmail.com), COMSATS University Islamabad, Pakistan",
                                  bg="#50E3C2", fg="#333333", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()
    def create_entry(self, text, default_val, row):
        label = tk.Label(self.input_frame, text=text, font=("Helvetica", 12, "bold italic"), fg="darkred", bg="white", anchor="e")
        label.grid(row=row*2, column=1, padx=10, pady=5, sticky="e")
        entry = tk.Entry(self.input_frame, font=("Helvetica", 12), fg="darkgreen", bg="white", width=15, bd=1, relief=tk.GROOVE)
        entry.insert(0, f"{default_val:.1f}")
        entry.grid(row=row*2, column=2, padx=10, pady=5, sticky="w")
        setattr(self, f'entry_{row}', entry)
    def get_entry_values(self):
        try:
            d1 = float(self.entry_1.get())
            d2 = float(self.entry_3.get())
            d3 = float(self.entry_5.get())
            d4 = float(self.entry_7.get())
            d5 = float(self.entry_9.get())  
            return d1, d2, d3, d4, d5
        except ValueError as ve:
            return None
    def calculate_y_a(self):
        values = self.get_entry_values()
        if values is None:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        y = 0.0
        y += (sqrt(sqrt(((d4/d2)+d3)))*(((self.G1C6+d1)/d2)/d2))
        y += (self.G2C2-(d5-(self.G2C0+d2)))
        y += (self.G3C5-((d2/(sqrt(self.G3C3)/d2))+(d1-d1)))
        y += (sqrt(((((d1+self.G4C5)+(self.G4C5+d1))+(self.G4C5*d2))/self.G4C1))*d2)
        self.a_output_text_a.delete(1.0, tk.END)
        self.a_output_text_a.insert(tk.END, f"{y:.2f}")
        self.a_output_text_a.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
    def calculate_b_b(self):
        values = self.get_entry_values()
        if values is None:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\Coral sand"
            filename = r"Coral.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=500)
            regressor= MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100,
                reg_lambda=0.1,
                gamma=1,
                max_depth=5
            ))
            model=regressor.fit(x_train, y_train)
            model= model.fit(x, y)
            y_pred=model.predict(x_train)
            y_pred=model.predict(x_test)
            input_data = np.array([d1, d2, d3, d4, d5]).reshape(1, -1)
            y_pred = model.predict(input_data)
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, f"{y_pred[0][0]:.2f}")
            self.b_output_text_b.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
        except FileNotFoundError:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Excel file not found")
        except ValueError as ve:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Invalid data format")
        except Exception as e:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Failure")
if __name__ == "__main__":
    root = tk.Tk()
    gui = RangeInputGUI(root)
    root.mainloop()