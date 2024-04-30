# EECE 7219: Pattern Recognition
# Final Software Assignment
# Kevin McKenzie
# 4/25/2024


from .model_3 import create_bayes_model
from .gui import BayesianNetworkGUI


if __name__ == "__main__":
    model = create_bayes_model()
    app = BayesianNetworkGUI(model)
    app.mainloop()
