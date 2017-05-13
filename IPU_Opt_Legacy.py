from __future__ import division
import pyomo
from pyomo.environ import *
import cplex
import logging

m = ConcreteModel()

#indexes
m.patient = RangeSet(1,8)
m.provider_type = Set(initialize = ['Image Tech', 'Surgeon', 'PT', 'Nutritionist'])
m.time = RangeSet(1,12)
m.test = m.patient * m.provider_type

#parameters
m.num_patients = Param(initialize = len(m.patient))
m.num_type_providers = Param(m.provider_type,initialize = {'Image Tech':2, 'Surgeon':2, 'PT':2, 'Nutritionist':2})
m.num_providers = Param(initialize = sum(m.num_type_providers.values()))
m.time_horizon = Param(initialize = len(m.time))
m.treatment_time = Param(m.provider_type, initialize = {'Image Tech':1, 'Surgeon':3, 'PT':3, 'Nutritionist':5})
m.admin_time = Param(m.provider_type, initialize = {'Image Tech':1, 'Surgeon':1, 'PT':1, 'Nutritionist':1})
m.patient_pathway = Param(m.test, initialize = {(1,'Image Tech'):0,(1,'Surgeon'):1, (1,'PT'):1, (1,'Nutritionist'):0, (2,'Image Tech'):0, (2,'Surgeon'):0, (2,'PT'):1, (2,'Nutritionist'):1, (3,'Image Tech'):1, (3,'Surgeon'):0, (3,'PT'):1, (3,'Nutritionist'):1, (4,'Image Tech'):0, (4,'Surgeon'):1, (4,'PT'):0, (4,'Nutritionist'):1, (5,'Image Tech'):0, (5,'Surgeon'):1, (5,'PT'):1, (5,'Nutritionist'):0, (6,'Image Tech'):0, (6,'Surgeon'):0, (6,'PT'):1, (6,'Nutritionist'):0, (7,'Image Tech'):0, (7,'Surgeon'):1, (7,'PT'):1, (7,'Nutritionist'):0, (8,'Image Tech'):0, (8,'Surgeon'):1, (8,'PT'):1, (8,'Nutritionist'):0})

m.num_rooms = Param(initialize = 8)

#variables
#T Variable
m.patient_finish = Var(m.patient, within = NonNegativeIntegers)
m.provider_finish = Var(m.provider_type, within = NonNegativeIntegers)

#X variable
m.patient_treated = Var(m.patient, m.provider_type, m.time, within = Binary)

#Y Variable
m.provider_treat_start = Var(m.patient, m.provider_type, m.time, within = Binary)
m.provider_treat_finish = Var(m.patient, m.provider_type, m.time, within = Binary)

m.last_patient = Var(within = NonNegativeIntegers)
m.over_time = Var(m.patient,m.provider_type,within = NonNegativeIntegers)

#Objective Function
def Objective_Function(model):
    return 0.2*summation(m.patient_finish) + 0.2*summation(m.provider_finish) + 0.6*summation(m.over_time)
m.Obj = Objective(rule = Objective_Function, sense = minimize)

#Constraints
def Patient_Begin_Treat(model,i,k):
    return m.provider_treat_start[i,k,1] == m.patient_treated[i,k,1]
m.New_Constraint = Constraint(m.patient, m.provider_type, rule = Patient_Begin_Treat)

def Patient_Begin_Treat_1(model,i,k,t):
	return m.provider_treat_start[i,k,t] <= m.patient_treated[i,k,t]
m.Patient_Begin_Treat_1 = Constraint(m.patient,m.provider_type,m.time, rule = Patient_Begin_Treat_1)

def Patient_Begin_Treat_2(model,i,k,t):
    if t > 1:
        return m.provider_treat_start[i,k,t] + m.patient_treated[i,k,t-1] <= 1
    else:
        return Constraint.Skip
m.Patient_Begin_Treat_2 = Constraint(m.patient,m.provider_type,m.time, rule = Patient_Begin_Treat_2)

def Patient_Begin_Treat_3(model,i,k,t):
    if t > 1:
        return -m.provider_treat_start[i,k,t] + m.patient_treated[i,k,t] - m.patient_treated[i,k,t-1] <= 0
    else:
        return Constraint.Skip
m.Patient_Begin_Treat_3 = Constraint(m.patient,m.provider_type,m.time, rule = Patient_Begin_Treat_3)

def Patient_Finish_Treat_1(model,i,k,t):
    if t > 1:
    	return m.provider_treat_finish[i,k,t] <= m.patient_treated[i,k,t-1]
    else:
        return Constraint.Skip
m.Patient_Finish_Treat_1 = Constraint(m.patient,m.provider_type,m.time, rule = Patient_Finish_Treat_1)

def Patient_Finish_Treat_2(model,i,k,t):    
    return m.provider_treat_finish[i,k,t] + m.patient_treated[i,k,t] <= 1
m.Patient_Finish_Treat_2 = Constraint(m.patient,m.provider_type,m.time, rule = Patient_Finish_Treat_2)

def Patient_Finish_Treat_3(model,i,k,t):
    if t > 1:
        return -m.provider_treat_finish[i,k,t] + m.patient_treated[i,k,t-1] - m.patient_treated[i,k,t] <= 0
    else:
        return Constraint.Skip
m.Patient_Finish_Treat_3 = Constraint(m.patient,m.provider_type,m.time, rule = Patient_Finish_Treat_3)

def Continous_Treatment(model,i,k,t):
    return sum(m.patient_treated[i,k,t2] for t2 in range(t,min(t+m.treatment_time[k], len(m.time)+1))) >= min(m.treatment_time[k], len(m.time)-t+1)*m.provider_treat_start[i,k,t]
m.Patient_Continous_Treatment = Constraint(m.patient, m.provider_type, m.time, rule = Continous_Treatment)

def Provider_Admin_Break(model,i,k,t):
    if t+m.admin_time[k] > len(m.time):
        return Constraint.Skip
    else:
        return sum(m.provider_treat_start[i2,k,t2] for t2 in range(t, t+m.admin_time[k]) for i2 in m.patient if i2 != i) <= 1-m.provider_treat_finish[i,k,t]
m.Provider_Admin_Break = Constraint(m.patient,m.provider_type, m.time, rule = Provider_Admin_Break)

def Max_Simultaneous_Treatment(model,k,t):
    return sum(m.patient_treated[i,k,t] for i in m.patient) <= m.num_type_providers[k]
m.Max_Simultaneous_Treatment = Constraint(m.provider_type, m.time, rule = Max_Simultaneous_Treatment)

def Time_Patient_Finish(model,i,k,t):
    return t*m.patient_treated[i,k,t] <= m.patient_finish[i]
m.Patient_Finish_Time = Constraint(m.patient, m.provider_type, m.time, rule = Time_Patient_Finish)

def Time_Provider_Finish(model,i,k,t):
    return t*m.patient_treated[i,k,t] <= m.provider_finish[k]
m.Provider_Finish_Time = Constraint(m.patient, m.provider_type, m.time, rule = Time_Provider_Finish)

def Patient_Pathway(model,i,k):
    if m.patient_pathway[i,k] == 0:
        return Constraint.Skip
    else:
        return sum(m.patient_pathway[i,k]*m.patient_treated[i,k,t] for t in m.time) >= 1
m.Patient_Path = Constraint(m.patient,m.provider_type, rule = Patient_Pathway)

def Room_Capacity(model,t):
    return sum(m.patient_treated[i,k,t] for i in m.patient for k in m.provider_type) <= m.num_rooms
m.Room_Capacity = Constraint(m.time, rule = Room_Capacity)

def Last_Patient(model,i):
    return m.patient_finish[i] <= m.last_patient
m.Last_Patient = Constraint(m.patient, rule = Last_Patient)

def Max_Treatment_Time(model,i,k):
    return sum(m.patient_pathway[i,k]*m.patient_treated[i,k,t] for t in m.time) + m.over_time[i,k]  == m.patient_pathway[i,k]*m.treatment_time[k]
m.Patient_Max_Treatment = Constraint(m.patient,m.provider_type, rule = Max_Treatment_Time)

solver = pyomo.opt.SolverFactory('cplex')
solver.options['mip cuts all'] = 2
solver.options['timelimit'] = 300
results = solver.solve(m, tee=True, keepfiles=True)
if (results.solver.status != pyomo.opt.SolverStatus.ok):
	logging.warning('Check solver not ok?')
if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
	logging.warning('Check solver optimality?')
m.solutions.load_from(results)
