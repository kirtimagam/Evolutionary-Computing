"""
evo.py: An evolutionary computing framework

"""
import random as rnd
import copy
from functools import reduce
import pickle
import csv
import time
import pandas as pd

class Environment:

    def __init__(self):
        self.pop = {}   # evaluation tuple ((name1, obj1), (name2, obj2)...) --> solution
                        # there are no duplicate in the population
        self.fitness = {} # name--> function
        self.agents = {} # name --> (operator, num_solution)

    def size(self):
        """ The number of solutions in the population """
        return len(self.pop)

    def add_fitness_criteria(self, name, f):
        """Every new solution is evaluated wrt
        each of the fitness criteria """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ Register an agent with the framework """
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        evaluation = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[evaluation] = sol

    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population """
        if self.size() == 0:
            return []
        else:
            solutions = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(solutions)) for _ in range(k)]

    def run_agent(self, name):
        """Invoke an agent against the population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        """ p = evaluation of solution: ((obj1, score1), (obj2, score2), ... )"""
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x,y: y-x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Environment._dominates(p,q)}

    def remove_dominated(self):
        """ Remove dominated solutions from the populations """
        nds = reduce(self._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}


    def evolve(self, n=1, dom = 100, status = 10000, sync = 1000):
        """ Run n random agents (default = 1) """

        # set prioritized best scores to infinity
        best_unwilling = float("inf")
        best_undersupport = float("inf")
        best_overallocation = float("inf")

        # instantiate list to hold best penalty scores
        best_lst = []

        # get time at beginning of evo call
        beg = time.time()

        # establish names of all agents
        agent_names = list(self.agents.keys())

        # iterate for n times
        for i in range(n):

            # check if time has been less than 10 minutes
            if time.time() - beg <= 600:
                pick = rnd.choice(agent_names)
                self.run_agent(pick)

                vals = []
                # for each penalty score in current iteration's solution
                for x in list(list(self.pop.items())[-1][0]):

                    # append each penalty score
                    vals.append(x[1])

                # only save scores that have a better unwilling score, and each of the
                # 4 other constraints we specified
                if vals[3] < best_unwilling and vals[1] < 6 and vals[0] < 5 and vals[4] < 15 and vals[2] < 5:

                    # only save if it is best undersupport or best overallocation
                    if vals[1] < best_undersupport or vals[0] < best_overallocation:

                        # print current best solution
                        print(vals)

                        # set new best prioritized scores
                        best_unwilling = vals[3]
                        best_undersupport = vals[3]
                        best_overallocation = vals[3]

                        # save current best penalty scores and actual ta assignments
                        best_lst = vals
                        this_sol = list(self.pop.items())[-1][1]

                if i % dom == 0:
                    self.remove_dominated()

                if i % status == 0:
                    self.remove_dominated()
                    print("Iteration: ", i)
                    print("Population size: ", self.size())
                    print(self)

                if i % sync == 0:

                    # load saved solutions and merge them
                    # into our population (leaving existing solutions
                    # unchanged)
                    try:
                        with open('solutions.dat', 'rb') as file:
                            loaded = pickle.load(file)
                            for eval, sol in loaded.items():
                                self.pop[eval] = sol
                    except Exception as e:
                        print(e)


                    # remove dominated solutions before saving to disk
                    self.remove_dominated()

                    # save the solutions
                    with open('solutions.dat', 'wb') as file:
                        pickle.dump(self.pop, file)

            # if time has been more than 10 minutes, stop iterating
            else:
                break

        # save last solutions ta assignments and titles
        titles = []
        vals = []
        for i in list(list(self.pop.items())[-1][0]):
            titles.append(i[0])
            vals.append(i[1])

        # insert group name into best solutions title and penalty scores
        titles.insert(0, "groupname")
        best_lst.insert(0, "group 9!")

        # make df for best solution's penalty scores
        gdf = pd.DataFrame([best_lst], columns = titles)

        # iterate through each assignment
        ta_counts = []
        for i in range(len(this_sol)):
            assignments = []
            for a in range(len(this_sol[i])):

                # if a ta is assigned, append their #lab assignment to list
                if this_sol[i][a] == 1:
                    assignments.append(a)

            # add ta assignment info to ta_counts list
            ta_counts.append(f"TA #{i} sections assignments: {assignments}")

        # make list of lists for labs
        lab_counts = []
        for i in range(0,17):
            lab_counts.append([])

        # fill lab ta assignments
        for i in range(len(this_sol)):
            #assignments = []
            for a in range(len(this_sol[i])):
                if this_sol[i][a] == 1:
                    lab_counts[a].append(i)

        # make list to save info for lab assignments
        lab_counts_d = []
        for i in range(len(lab_counts)):
            lab_counts_d.append(f"Lab #{i} TA assignments: {lab_counts[i]}")

        # make df for lab and ta assignments
        gdf_labassign = pd.DataFrame(lab_counts_d)
        gdf_TAassign = pd.DataFrame(ta_counts)

        # print and save best solution
        print(gdf_labassign)
        gdf_labassign.to_csv("gdf_labassign_4.csv")
        print(gdf_TAassign)
        gdf_TAassign.to_csv("gdf_TAassign_4.csv")
        print(gdf)
        gdf.to_csv("scores_4.csv")

        # save each non-dominated Pareto-optimal solution:
        # iterate through each assignment
        biglst = []
        for i in range(len(list(self.pop.items()))):
            vals = []
            for x in list(list(self.pop.items())[i][0]):

                # append penalty score for each non dominated solution
                vals.append(x[1])

            # insert groupname column
            vals.insert(0, "group 9!")

            # append each solution
            biglst.append(vals)

        # make df of all solutions
        df = pd.DataFrame(biglst, columns=titles)

        # save to csv and print all non dominated solutions
        df.to_csv("sol_3.csv")
        print(df)

        # remove non dominated solutions
        self.remove_dominated()

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            rslt += str(dict(eval))+":\t"+str(sol)+"\n"
        return rslt