from skopt.space import Integer
from skopt import Optimizer, dummy_minimize
from scipy.optimize import minimize
import numpy as np
import time

def gradient_opt(transferEnvironment):
    max_thread, count = transferEnvironment.action_space.n, 0
    soft_limit, least_cost = max_thread, 0
    values = []
    ccs = [2]
    theta = 0
    # timer320s=time.time()
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

        gradient = (values[-1] - values[-2])/2
        gradient_change = np.abs(gradient/values[-2])

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


def gradient_opt_fast(transferEnvironment):
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
            dist = max(1, np.abs(ccs[-1] - ccs[-2]))
            if ccs[-1]>ccs[-2]:
                gradient = (values[-1] - values[-2])/dist
            else:
                gradient = (values[-2] - values[-1])/dist

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


def bayes_optimizer(transferEnvironment,configurations):
    limit_obs, count = 25, 0
    max_thread = transferEnvironment.action_space.n
    iterations = configurations["bayes"]["num_of_exp"]
    search_space  = [
            Integer(1, max_thread), # Concurrency
        ]
    params = []
    optimizer = Optimizer(
        dimensions=search_space,
        base_estimator="GP", #[GP, RF, ET, GBRT],
        acq_func="gp_hedge", # [LCB, EI, PI, gp_hedge]
        acq_optimizer="auto", #[sampling, lbfgs, auto]
        n_random_starts=configurations["bayes"]["initial_run"],
        model_queue_size= limit_obs,
        # acq_func_kwargs= {},
        # acq_optimizer_kwargs={}
    )
    # timer320s=time.time()
    while True:
        count +=1
        if len(optimizer.yi) > limit_obs:
            optimizer.yi = optimizer.yi[-limit_obs:]
            optimizer.Xi = optimizer.Xi[-limit_obs:]


        transferEnvironment.transferClassObject.log.info("Iteration {0} Starts ...".format(count))

        t1 = time.time()
        res = optimizer.run(func=transferEnvironment.bayes_step, n_iter=1)
        t2 = time.time()


        transferEnvironment.transferClassObject.log.info("Iteration {0} Ends, Took {3} Seconds. Best Params: {1} and Score: {2}.".format(
                count, res.x, res.fun, np.round(t2-t1, 2)))

        last_value = optimizer.yi[-1]
        if last_value == 10 ** 10:
            transferEnvironment.transferClassObject.log.info("Bayseian Optimizer Exits ...")
            break

        cc = optimizer.Xi[-1][0]
        if iterations < 1:
            reset = False
            if (last_value > 0) and (cc < max_thread):
                max_thread = max(cc, 2)
                reset = True

            if (last_value < 0) and (cc == max_thread) and (cc < configurations["thread_limit"]):
                max_thread = min(cc+5, configurations["thread_limit"])
                reset = True

            if reset:
                search_space[0] = Integer(1, max_thread)
                optimizer = Optimizer(
                    dimensions=search_space,
                    n_initial_points=configurations["bayes"]["initial_run"],
                    acq_optimizer="lbfgs",
                    model_queue_size= limit_obs
                )
        # if (timer320s + 300 <= time.time()):
        #     transferEnvironment.transferClassObject.file_incomplete.value=0
        if iterations == count:
            transferEnvironment.transferClassObject.log.info("Best parameters: {0} and score: {1}".format(res.x, res.fun))
            params = res.x
            break

    return params
