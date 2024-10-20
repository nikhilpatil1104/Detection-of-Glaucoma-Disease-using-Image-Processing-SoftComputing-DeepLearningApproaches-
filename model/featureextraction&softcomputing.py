import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

# Fuzzy variables
cdr = ctrl.Antecedent(np.arange(0, 1, 0.01), 'cdr')  # Cup-to-Disc Ratio
risk = ctrl.Consequent(np.arange(0, 1, 0.01), 'risk')

# Membership functions
cdr['low'] = fuzz.trimf(cdr.universe, [0, 0, 0.5])
cdr['medium'] = fuzz.trimf(cdr.universe, [0.4, 0.6, 0.8])
cdr['high'] = fuzz.trimf(cdr.universe, [0.7, 1, 1])

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 0.3])
risk['medium'] = fuzz.trimf(risk.universe, [0.2, 0.5, 0.8])
risk['high'] = fuzz.trimf(risk.universe, [0.7, 1, 1])

# Define fuzzy rules
rule1 = ctrl.Rule(cdr['high'], risk['high'])
rule2 = ctrl.Rule(cdr['medium'], risk['medium'])
rule3 = ctrl.Rule(cdr['low'], risk['low'])

# Control system and simulation
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# Example input
risk_sim.input['cdr'] = 0.65  # Example CDR value
risk_sim.compute()
print(f'Risk of Glaucoma: {risk_sim.output["risk"]:.2f}')
