# EECE 7219: Pattern Recognition
# Final Software Assignment
# Kevin McKenzie
# 4/25/2024


from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def create_bayes_model():

    # define structure of model
    model = BayesianNetwork([('A', 'X'), ('B', 'X'), ('X', 'C'), ('X', 'D')])

    # define conditional probability distributions (CPD)
    cpd_a = TabularCPD(variable='A', variable_card=4,
                       values=[[0.25], [0.25], [0.25], [0.25]],
                       state_names={'A': ['a1', 'a2', 'a3', 'a4']})

    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.6], [0.4]],
                       state_names={'B': ['b1', 'b2']})

    cpd_x = TabularCPD(variable='X', variable_card=2,
                       values=[[0.5, 0.7, 0.6, 0.8, 0.4, 0.1, 0.2, 0.3],
                               [0.5, 0.3, 0.4, 0.2, 0.6, 0.9, 0.8, 0.7]],
                       evidence=['A', 'B'],
                       evidence_card=[4, 2],
                       state_names={'X': ['x1', 'x2'], 'A': ['a1', 'a2', 'a3', 'a4'], 'B': ['b1', 'b2']})

    cpd_c = TabularCPD(variable='C', variable_card=3,
                       values=[[0.6, 0.2], [0.2, 0.3], [0.2, 0.5]],
                       evidence=['X'],
                       evidence_card=[2],
                       state_names={'C': ['c1', 'c2', 'c3'], 'X': ['x1', 'x2']})

    cpd_d = TabularCPD(variable='D', variable_card=2,
                       values=[[0.3, 0.6], [0.7, 0.4]],
                       evidence=['X'],
                       evidence_card=[2],
                       state_names={'D': ['d1', 'd2'], 'X': ['x1', 'x2']})


    model.add_cpds(cpd_a, cpd_b, cpd_x, cpd_c, cpd_d)
    assert model.check_model()

    return model
