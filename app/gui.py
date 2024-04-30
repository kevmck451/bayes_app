# EECE 7219: Pattern Recognition
# Final Software Assignment
# Kevin McKenzie
# 4/25/2024


import tkinter as tk
from tkinter import messagebox
from pgmpy.inference import VariableElimination


class BayesianNetworkGUI(tk.Tk):

    def __init__(self, model):
        super().__init__()
        self.title("Bayesian Network Inference")
        self.model = model
        self.inference = VariableElimination(self.model)
        self.state_names = {
            'A': ['Winter', 'Spring', 'Summer', 'Autumn'],
            'B': ['North Atlantic', 'South Atlantic'],
            'C': ['Light', 'Medium', 'Dark'],
            'D': ['Wide', 'Thin']
        }
        self.create_widgets()


    def create_widgets(self):

        instructions = "App Instructions: \n" \
                       "Select the known conditions and leave at least one field as 'Unknown' to perform inference. " \
                       "The results will display the probability distributions for the unknown variables."
        tk.Label(self, text=instructions, wraplength=400, justify="left").pack(pady=(10, 10))

        # labels and buttons for each variable
        self.vars = {
            'A': ('Season', ['Winter', 'Spring', 'Summer', 'Autumn']),
            'B': ('Locale', ['North Atlantic', 'South Atlantic']),
            'C': ('Lightness', ['Light', 'Medium', 'Dark']),
            'D': ('Thickness', ['Wide', 'Thin'])
        }
        self.var_values = {var: tk.StringVar(value='Unknown') for var in self.vars}
        
        for i, (var, (label_text, options)) in enumerate(self.vars.items(), start=1):
            frame = tk.LabelFrame(self, text=label_text)
            frame.pack(padx=10, pady=5, fill="x")
            options.append('Unknown')  # Add 'Unknown' option
            for option in options:
                tk.Radiobutton(frame, text=option, variable=self.var_values[var], value=option).pack(side="left")

        # calculate button
        calculate_btn = tk.Button(self, text="Calculate", command=self.perform_inference)
        calculate_btn.pack(pady=(5, 20))

        # Result display
        self.result_display = tk.Text(self, height=20, width=50)
        self.result_display.pack()


    def perform_inference(self):

        # prep evidence dictionary for known conditions
        evidence = {}
        for var, value in self.var_values.items():
            if value.get() != 'Unknown':
                # Find index of text value and use technical value
                index = self.vars[var][1].index(value.get())
                evidence[var] = self.model.get_cpds(var).state_names[var][index]

        query_vars = [var for var in self.vars if self.var_values[var].get() == 'Unknown']
        
        # if all variables set to 'unknown', make popup
        if not evidence or not query_vars:
            messagebox.showerror("Error", "Please specify at least one known and one unknown variable.")
            return

        # perform inference
        try:
            result = self.inference.query(variables=query_vars, evidence=evidence, joint=False)
            self.result_display.delete('1.0', tk.END)
            for var in query_vars:
                distribution = result[var]
                # Construct the string to display each state's probability
                for state_index, probability in enumerate(distribution.values):
                    state_name = self.state_names[var][state_index]
                    self.result_display.insert(tk.END, f"{var} ({state_name}): {probability:.4f}\n")
                self.result_display.insert(tk.END, "\n")
        
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))

