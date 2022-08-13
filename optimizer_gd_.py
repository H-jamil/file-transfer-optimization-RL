from skopt.space import Integer
from skopt import Optimizer, dummy_minimize
from scipy.optimize import minimize
import numpy as np
import time

def gradient_opt_(transferEnvironment):
    max_thread, count = transferEnvironment.action_space.n, 0
    soft_limit, least_cost = max_thread, 0
    values = []
    ccs = [2]
    theta = 0
    while True:
        state,score,done,_=transferEnvironment.step(ccs[-1]-1)
        values.append(score)
        if done:
            transferEnvironment.transferClassObject.log.info("GD Optimizer Exits ...")
            break
        if values[-1] < least_cost:
          least_cost = values[-1]
          soft_limit = min(ccs[-1]+10, max_thread)
        state,score,done,_=transferEnvironment.step(ccs[-1]+1)
        values.append(score)
        if done:
            transferEnvironment.transferClassObject.log.info("GD Optimizer Exits ...")
            break
        if values[-1] < least_cost:
          least_cost = values[-1]
          soft_limit = min(ccs[-1]+10, max_thread)

        count += 2
        if len(values)< 6:
          gradient = (values[-1] - values[-2])/2
          gradient_change = np.abs(gradient/values[-2])
        else:
          gradient = (np.mean(values[-5:]) - np.mean(values[-6:]))/2
          gradient_change = np.abs(gradient/np.mean(values[-6:]))

        if gradient>0:
            if theta <= 0:
                theta -= 1
            else:
                theta = -1

        else:
            if theta >= 0:
                theta += 1
            else:
                theta = 1

        update_cc = int(theta * np.ceil(ccs[-1] * gradient_change))
        next_cc = min(max(ccs[-1] + update_cc, 2), soft_limit-1)
        transferEnvironment.transferClassObject.log.info("Gradient: {0}, Gredient Change: {1}, Theta: {2}, Previous CC: {3}, Choosen CC: {4}".format(gradient, gradient_change, theta, ccs[-1], next_cc))
        ccs.append(next_cc)
        # if (timer320s + 300 <= time.time()):
        #     transferEnvironment.transferClassObject.file_incomplete.value=0
    return ccs


def gradient_opt_fast_(transferEnvironment):
    max_thread, count = transferEnvironment.action_space.n, 0
    soft_limit, least_cost = max_thread, 0
    values = []
    ccs = [1]
    theta = 0
    # timer320s=time.time()
    while True:
        state,score,done,_=transferEnvironment.step(ccs[-1])
        values.append(score)
        if done:
            transferEnvironment.transferClassObject.log.info("GD_Fast Optimizer Exits ...")
            break

        if values[-1] < least_cost:
            least_cost = values[-1]
            soft_limit = min(ccs[-1]+10, max_thread)

        if len(ccs) == 1:
            ccs.append(2)

        else:
            if len(values)< 6:
              dist = max(1, np.abs(ccs[-1] - ccs[-2]))
              if ccs[-1]>ccs[-2]:
                gradient = (values[-1] - values[-2])/dist
              else:
                gradient = (values[-2] - values[-1])/dist
            else:
              dist = max(1, np.abs(ccs[-1] - ccs[-2]))
              if ccs[-1]>ccs[-2]:
                gradient = (np.mean(values[-5:]) - np.mean(values[-6:]))/dist
              else:
                gradient = (np.mean(values[-6:]) - np.mean(values[-5:]))/dist

            if values[-2] !=0:
                gradient_change = np.abs(gradient/values[-2])
            else:
                gradient_change = np.abs(gradient)

            if gradient>0:
                if theta <= 0:
                    theta -= 1
                else:
                    theta = -1

            else:
                if theta >= 0:
                    theta += 1
                else:
                    theta = 1

            update_cc = int(theta * np.ceil(ccs[-1] * gradient_change))
            next_cc = min(max(ccs[-1] + update_cc, 2), soft_limit)
            transferEnvironment.transferClassObject.log.info("Gradient_Fast: {0}, Gredient_Fast Change: {1}, Theta: {2}, Previous CC: {3}, Choosen CC: {4}".format(gradient, gradient_change, theta, ccs[-1], next_cc))
            ccs.append(next_cc)
            # if (timer320s + 300 <= time.time()):
            #     transferEnvironment.transferClassObject.file_incomplete.value=0
    return ccs
